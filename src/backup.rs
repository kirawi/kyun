use std::fs::Metadata;
use std::io::{Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow, bail};
use filetime::FileTime;
use serde::Deserialize;

use crate::faccess;

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BackupKind {
    None,
    Copy,
    #[default]
    Auto,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct BackupConfig {
    pub kind: BackupKind,
    pub directories: Vec<PathBuf>,
    pub extension: String,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            kind: BackupKind::Auto,
            directories: vec![PathBuf::from(".")],
            extension: "bck".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackupMethod {
    Copy,
    Rename,
}

#[derive(Debug)]
pub struct Backup {
    pub method: BackupMethod,
    pub path: PathBuf,
}

impl Backup {
    pub async fn create(
        path: &Path,
        requested_path: &Path,
        is_symlink: bool,
        cwd: &Path,
        config: &BackupConfig,
    ) -> anyhow::Result<Self> {
        // This fn won't be called if it's none or it doesn't exist
        if config.kind == BackupKind::None || !path.exists() {
            unreachable!();
        }

        let mut method = if config.kind == BackupKind::Copy {
            BackupMethod::Copy
        } else {
            BackupMethod::Rename
        };

        // Do not rename iff:
        // - it is a hardlink
        // - it is a symlink
        // - we don't have file create perms for the dir
        if method == BackupMethod::Rename {
            // Conservatively assume it is a hardlink if we can't read metadata
            let is_hardlink = {
                let p_ = path.to_owned();
                tokio::task::spawn_blocking(move || faccess::hardlink_count(&p_).unwrap_or(2))
                    .await?
                    > 1
            };

            // We must manually copy the file into a backup because `move` would destroy links.
            if is_hardlink || is_symlink {
                method = BackupMethod::Copy;
            } else {
                // We are probing the process's permissions here before actually creating a backup.
                let metadata = tokio::fs::metadata(path).await?;
                let permissions = metadata.permissions();
                let parent = path.parent().unwrap_or_else(|| Path::new("."));
                let mut builder = tempfile::Builder::new();
                builder.permissions(permissions);

                if let Ok(file) = builder.tempfile_in(parent) {
                    // Check if we have perms to set perms
                    #[cfg(unix)]
                    {
                        use std::os::{fd::AsFd, unix::fs::MetadataExt};

                        let _ = faccess::fchown(
                            file.as_file().as_fd(),
                            Some(metadata.uid()),
                            Some(metadata.gid()),
                        );
                        if let Ok(temporary_metadata) = file.as_file().metadata() {
                            // Check if file perms were updated. If not, we assume the process does not have permissions to
                            // modify perms of this file and must instead copy.
                            if metadata.uid() != temporary_metadata.uid()
                                || metadata.gid() != temporary_metadata.gid()
                                || metadata.permissions() != temporary_metadata.permissions()
                            {
                                method = BackupMethod::Copy;
                            }
                        } else {
                            method = BackupMethod::Copy;
                        }
                    }

                    #[cfg(not(unix))]
                    if faccess::copy_metadata(path, file.path()).is_err() {
                        method = BackupMethod::Copy;
                    }
                } else {
                    method = BackupMethod::Copy;
                }
            }
        }

        if config.directories.is_empty() {
            bail!("no backup directories are configured");
        }

        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let mut directories: Vec<_> = config
            .directories
            .iter()
            .map(|directory| {
                if directory == Path::new(".") {
                    parent.to_owned()
                } else if directory.is_absolute() {
                    directory.clone()
                } else {
                    cwd.join(directory)
                }
            })
            .collect();

        if !directories.iter().any(|directory| directory.is_dir()) {
            let directory = directories.last().unwrap();
            tokio::fs::create_dir_all(directory)
                .await
                .with_context(|| {
                    format!(
                        "could not create backup directory '{}'",
                        directory.display()
                    )
                })?;
        }

        directories.retain(|directory| directory.is_dir());

        // Look for valid backup directory
        // Check if:
        // - directory is not writable
        // - path is a directory
        // - path exists
        let mut escaped_p = requested_path.to_string_lossy().into_owned();
        escaped_p = escaped_p.replace('/', "%");
        if cfg!(windows) {
            escaped_p = escaped_p.replace(['\\', ':'], "%");
        }

        let mut last_error = None;
        'outer: for directory in directories {
            let ext = config.extension.as_str();

            // If the backup file already exists, we'll try to add a number before the extension
            // until we're done
            // NOTE: u8 since if we need more than 256, there might be an issue
            let mut number = 0_u8;
            loop {
                // NOTE: `escaped_p` will make dot files appear to be extensions, so we need to append
                let suffix = if number == 0 {
                    format!(".{ext}")
                } else {
                    format!(".{number}.{ext}")
                };
                let candidate = directory.join(format!("{escaped_p}{suffix}"));
                match tokio::fs::symlink_metadata(&candidate).await {
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Ok(_) | Err(_) => {
                        if number == u8::MAX {
                            continue 'outer;
                        }
                        number += 1;
                        continue;
                    }
                }

                if method == BackupMethod::Rename {
                    if tokio::fs::rename(path, &candidate).await.is_ok() {
                        return Ok(Self {
                            method: BackupMethod::Rename,
                            path: candidate,
                        });
                    }
                    method = BackupMethod::Copy;
                }

                match Self::create_copy(path, &candidate).await {
                    Ok(backup) => return Ok(backup),
                    Err(error)
                        if error.downcast_ref::<std::io::Error>().is_some_and(|error| {
                            error.kind() == std::io::ErrorKind::AlreadyExists
                        }) =>
                    {
                        if number == u8::MAX {
                            continue 'outer;
                        }
                        number += 1;
                    }
                    Err(error) => {
                        let _ = tokio::fs::remove_file(&candidate).await;
                        last_error = Some(error);
                        break;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("could not write into a backup directory")))
    }

    async fn create_copy(path: &Path, backup_path: &Path) -> anyhow::Result<Self> {
        let metadata = tokio::fs::metadata(path).await?;
        copy_file_contents(path, backup_path, true).await?;

        // Ownership, permissions, extended attributes, and times are all synchronous metadata operations;
        // keep them off the runtime workers.
        let path = path.to_owned();
        let backup_path = backup_path.to_owned();
        tokio::task::spawn_blocking(move || -> anyhow::Result<Self> {
            #[cfg(unix)]
            {
                use std::os::{
                    fd::AsFd,
                    unix::fs::{MetadataExt, PermissionsExt},
                };

                let backup_file = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&backup_path)?;
                let backup_metadata = backup_file.metadata()?;
                let original_gid = metadata.gid();
                let backup_gid = backup_metadata.gid();

                let mut permissions = metadata.permissions();
                permissions.set_mode(permissions.mode() & 0o0777); // Strip s-bit

                // If chown fails, set the protection bits for the group the same as the perm bits for others
                if original_gid != backup_gid
                    && faccess::fchown(backup_file.as_fd(), None, Some(original_gid)).is_err()
                {
                    let new_permissions =
                        (permissions.mode() & 0o0707) | ((permissions.mode() & 0o07) << 3);
                    permissions.set_mode(new_permissions);
                }

                #[cfg(target_os = "macos")]
                {
                    use std::fs::{File, FileTimes};
                    use std::os::macos::fs::FileTimesExt;

                    let backup_file = File::options().write(true).open(&backup_path)?;
                    let times = FileTimes::new().set_created(metadata.created()?);
                    backup_file.set_times(times)?;
                }

                std::fs::set_permissions(&backup_path, permissions)?;
                if let Err(error) = faccess::copy_xattr(&path, &backup_path) {
                    let unsupported =
                        rustix::io::Errno::from_io_error(&error).is_some_and(|errno| {
                            errno == rustix::io::Errno::NOTSUP
                                || errno == rustix::io::Errno::OPNOTSUPP
                        });
                    if !unsupported {
                        return Err(error.into());
                    }
                }
            }

            #[cfg(windows)]
            {
                faccess::copy_metadata(&path, &backup_path)?;
            }

            let access_time = FileTime::from_last_access_time(&metadata);
            let modification_time = FileTime::from_last_modification_time(&metadata);
            filetime::set_file_times(&backup_path, access_time, modification_time)?;
            Ok(Self {
                method: BackupMethod::Copy,
                path: backup_path,
            })
        })
        .await?
    }

    pub async fn restore(&self, path: &Path, metadata: &Metadata) -> anyhow::Result<()> {
        match self.method {
            BackupMethod::Copy => {
                // Restore backup
                copy_file_contents(&self.path, path, false).await?;
                apply_restored_metadata(&self.path, path, metadata).await?;
            }
            BackupMethod::Rename => {
                // restore backup
                tokio::fs::rename(&self.path, path).await?;
                // Essentially no modification to the content had been done; the moved file already carries
                // its own ownership and attributes (the backup path no longer exists).
                let access_time = FileTime::from_last_access_time(metadata);
                let modification_time = FileTime::from_last_modification_time(metadata);
                filetime::set_file_times(path, access_time, modification_time)?;
            }
        }
        Ok(())
    }
}

/// Applies the original file's ownership, permissions, extended attributes, and times to a restored path.
/// All synchronous metadata work runs on the blocking pool.
async fn apply_restored_metadata(
    backup_path: &Path,
    path: &Path,
    metadata: &Metadata,
) -> anyhow::Result<()> {
    let backup_path = backup_path.to_owned();
    let path = path.to_owned();
    let metadata = metadata.clone();
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;

            faccess::chown(&path, Some(metadata.uid()), Some(metadata.gid()))?;
            std::fs::set_permissions(&path, metadata.permissions())?;
            if let Err(error) = faccess::copy_xattr(&backup_path, &path) {
                let unsupported = rustix::io::Errno::from_io_error(&error).is_some_and(|errno| {
                    errno == rustix::io::Errno::NOTSUP || errno == rustix::io::Errno::OPNOTSUPP
                });
                if !unsupported {
                    return Err(error.into());
                }
            }
        }

        #[cfg(windows)]
        faccess::copy_metadata(&backup_path, &path)?;

        let access_time = FileTime::from_last_access_time(&metadata);
        let modification_time = FileTime::from_last_modification_time(&metadata);
        filetime::set_file_times(&path, access_time, modification_time)?;
        Ok(())
    })
    .await?
}

async fn copy_file_contents(
    source: &Path,
    destination: &Path,
    create_new: bool,
) -> anyhow::Result<()> {
    let source = source.to_owned();
    let destination = destination.to_owned();
    tokio::task::spawn_blocking(move || {
        let mut source = std::fs::File::open(source)?;
        let mut destination = std::fs::OpenOptions::new()
            .create(!create_new)
            .create_new(create_new)
            .read(true)
            .write(true)
            .open(destination)?;
        destination.set_len(0)?;

        #[cfg(any(target_os = "android", target_os = "linux"))]
        if rustix::fs::ioctl_ficlone(&destination, &source).is_ok() {
            destination.sync_all()?;
            return Ok::<_, std::io::Error>(());
        }

        source.seek(SeekFrom::Start(0))?;
        destination.seek(SeekFrom::Start(0))?;
        std::io::copy(&mut source, &mut destination)?;
        destination.sync_all()
    })
    .await??;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn set_test_xattr(path: &Path, name: &std::ffi::CStr) -> bool {
        match rustix::fs::setxattr(path, name, b"metadata", rustix::fs::XattrFlags::empty()) {
            Ok(()) => true,
            Err(error)
                if error == rustix::io::Errno::NOTSUP
                    || error == rustix::io::Errno::OPNOTSUPP
                    || error == rustix::io::Errno::PERM
                    || error == rustix::io::Errno::ACCESS =>
            {
                false
            }
            Err(error) => panic!("could not set test xattr: {error}"),
        }
    }

    #[tokio::test]
    async fn copy_backup_keeps_the_original_path_and_can_restore_it() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let backup_directory = directory.path().join("backups");
        tokio::fs::create_dir(&backup_directory).await.unwrap();
        let path = directory.path().join("output.txt");
        tokio::fs::write(&path, "old").await.unwrap();
        let metadata = tokio::fs::metadata(&path).await.unwrap();
        let config = BackupConfig {
            kind: BackupKind::Copy,
            directories: vec![backup_directory],
            extension: "bck".into(),
        };
        let cwd = tokio::fs::canonicalize(".").await.unwrap();

        let backup = Backup::create(&path, &path, false, &cwd, &config)
            .await
            .unwrap();

        assert_eq!(backup.method, BackupMethod::Copy);
        assert_eq!(tokio::fs::read_to_string(&path).await.unwrap(), "old");
        tokio::fs::write(&path, "partial").await.unwrap();
        backup.restore(&path, &metadata).await.unwrap();
        assert_eq!(tokio::fs::read_to_string(&path).await.unwrap(), "old");
        tokio::fs::remove_file(backup.path).await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn copy_backup_preserves_unix_metadata() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let backup_directory = directory.path().join("backups");
        tokio::fs::create_dir(&backup_directory).await.unwrap();
        let path = directory.path().join("output.txt");
        tokio::fs::write(&path, "old").await.unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        let old_time = FileTime::from_unix_time(1_000_000, 0);
        filetime::set_file_times(&path, old_time, old_time).unwrap();
        let xattr_name = c"user.kyun-copy-test";
        let xattr_supported = set_test_xattr(&path, xattr_name);
        let config = BackupConfig {
            kind: BackupKind::Copy,
            directories: vec![backup_directory],
            extension: "bck".into(),
        };
        let cwd = tokio::fs::canonicalize(".").await.unwrap();

        let backup = Backup::create(&path, &path, false, &cwd, &config)
            .await
            .unwrap();

        let metadata = tokio::fs::metadata(&backup.path).await.unwrap();
        assert_eq!(metadata.mode() & 0o7777, 0o640);
        let original_metadata = tokio::fs::metadata(&path).await.unwrap();
        assert_eq!(metadata.uid(), original_metadata.uid());
        assert_eq!(metadata.gid(), original_metadata.gid());
        assert_eq!(FileTime::from_last_access_time(&metadata), old_time);
        assert_eq!(FileTime::from_last_modification_time(&metadata), old_time);
        if xattr_supported {
            let mut value = [0; 8];
            let length = rustix::fs::getxattr(&backup.path, xattr_name, &mut value).unwrap();
            assert_eq!(&value[..length], b"metadata");
        }
    }

    #[tokio::test]
    async fn automatic_backup_renames_and_can_restore_an_ordinary_file() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let backup_directory = directory.path().join("backups");
        tokio::fs::create_dir(&backup_directory).await.unwrap();
        let path = directory.path().join("output.txt");
        tokio::fs::write(&path, "old").await.unwrap();
        let metadata = tokio::fs::metadata(&path).await.unwrap();
        let config = BackupConfig {
            kind: BackupKind::Auto,
            directories: vec![backup_directory],
            extension: "bck".into(),
        };
        let cwd = tokio::fs::canonicalize(".").await.unwrap();

        let backup = Backup::create(&path, &path, false, &cwd, &config)
            .await
            .unwrap();

        assert_eq!(backup.method, BackupMethod::Rename);
        assert!(!path.exists());
        tokio::fs::write(&path, "partial").await.unwrap();
        backup.restore(&path, &metadata).await.unwrap();
        assert_eq!(tokio::fs::read_to_string(&path).await.unwrap(), "old");
        assert!(!backup.path.exists());
    }

    #[tokio::test]
    async fn backup_path_uses_the_requested_name_extension_and_numbering() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let backup_directory = directory.path().join("backups");
        tokio::fs::create_dir(&backup_directory).await.unwrap();
        let path = directory.path().join("source.txt");
        tokio::fs::write(&path, "old").await.unwrap();
        let collision = backup_directory.join("nested%source.txt.save");
        tokio::fs::write(&collision, "keep").await.unwrap();
        let config = BackupConfig {
            kind: BackupKind::Copy,
            directories: vec![backup_directory],
            extension: "save".into(),
        };
        let cwd = tokio::fs::canonicalize(".").await.unwrap();

        let backup = Backup::create(&path, Path::new("nested/source.txt"), false, &cwd, &config)
            .await
            .unwrap();

        assert_eq!(backup.path.file_name().unwrap(), "nested%source.txt.1.save");
        assert_eq!(tokio::fs::read_to_string(collision).await.unwrap(), "keep");
        tokio::fs::remove_file(backup.path).await.unwrap();
    }

    #[tokio::test]
    async fn backup_directories_are_tried_in_order() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let first = directory.path().join("first");
        let second = directory.path().join("second");
        tokio::fs::create_dir(&first).await.unwrap();
        tokio::fs::create_dir(&second).await.unwrap();
        for number in 0..=u8::MAX {
            let name = if number == 0 {
                "source.bck".into()
            } else {
                format!("source.{number}.bck")
            };
            tokio::fs::write(first.join(name), "keep").await.unwrap();
        }
        let path = directory.path().join("source.txt");
        tokio::fs::write(&path, "old").await.unwrap();
        let config = BackupConfig {
            kind: BackupKind::Copy,
            directories: vec![first, second.clone()],
            extension: "bck".into(),
        };
        let cwd = tokio::fs::canonicalize(".").await.unwrap();

        let backup = Backup::create(&path, Path::new("source"), false, &cwd, &config)
            .await
            .unwrap();

        assert_eq!(backup.path.parent().unwrap(), second);
        tokio::fs::remove_file(backup.path).await.unwrap();
    }

    #[tokio::test]
    async fn final_backup_directory_is_created_when_none_exist() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let first = directory.path().join("missing-first");
        let final_directory = directory.path().join("missing-final");
        let path = directory.path().join("source.txt");
        tokio::fs::write(&path, "old").await.unwrap();
        let config = BackupConfig {
            kind: BackupKind::Copy,
            directories: vec![first, final_directory.clone()],
            extension: "bck".into(),
        };
        let cwd = tokio::fs::canonicalize(".").await.unwrap();

        let backup = Backup::create(&path, Path::new("source"), false, &cwd, &config)
            .await
            .unwrap();

        assert_eq!(backup.path.parent().unwrap(), final_directory);
        tokio::fs::remove_file(backup.path).await.unwrap();
    }

    #[tokio::test]
    async fn dot_backup_directory_means_the_files_directory() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("source.txt");
        tokio::fs::write(&path, "old").await.unwrap();
        let config = BackupConfig {
            kind: BackupKind::Copy,
            ..BackupConfig::default()
        };
        let cwd = tokio::fs::canonicalize(".").await.unwrap();

        let backup = Backup::create(&path, Path::new("source"), false, &cwd, &config)
            .await
            .unwrap();

        assert_eq!(backup.path.parent(), path.parent());
        tokio::fs::remove_file(backup.path).await.unwrap();
    }

    #[tokio::test]
    async fn failed_restoration_retains_the_copy_backup() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let source_directory = directory.path().join("source");
        let backup_directory = directory.path().join("backups");
        tokio::fs::create_dir(&source_directory).await.unwrap();
        tokio::fs::create_dir(&backup_directory).await.unwrap();
        let path = source_directory.join("output.txt");
        tokio::fs::write(&path, "old").await.unwrap();
        let metadata = tokio::fs::metadata(&path).await.unwrap();
        let config = BackupConfig {
            kind: BackupKind::Copy,
            directories: vec![backup_directory],
            extension: "bck".into(),
        };
        let cwd = tokio::fs::canonicalize(".").await.unwrap();
        let backup = Backup::create(&path, &path, false, &cwd, &config)
            .await
            .unwrap();
        tokio::fs::remove_file(&path).await.unwrap();
        tokio::fs::remove_dir(source_directory).await.unwrap();

        assert!(backup.restore(&path, &metadata).await.is_err());
        assert!(backup.path.exists());
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn automatic_backup_falls_back_to_copy_across_filesystems() {
        use std::os::unix::fs::MetadataExt;

        let source_directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let Ok(backup_directory) = tempfile::tempdir_in("/dev/shm") else {
            return;
        };
        if source_directory.path().metadata().unwrap().dev()
            == backup_directory.path().metadata().unwrap().dev()
        {
            return;
        }
        let path = source_directory.path().join("output.txt");
        tokio::fs::write(&path, "old").await.unwrap();
        let config = BackupConfig {
            kind: BackupKind::Auto,
            directories: vec![backup_directory.path().to_owned()],
            extension: "bck".into(),
        };
        let cwd = tokio::fs::canonicalize(".").await.unwrap();

        let backup = Backup::create(&path, &path, false, &cwd, &config)
            .await
            .unwrap();

        assert_eq!(backup.method, BackupMethod::Copy);
        assert!(path.exists());
        assert_eq!(
            tokio::fs::read_to_string(&backup.path).await.unwrap(),
            "old"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn backup_collision_detection_does_not_follow_symlinks() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let backup_directory = directory.path().join("backups");
        tokio::fs::create_dir(&backup_directory).await.unwrap();
        let path = directory.path().join("source.txt");
        tokio::fs::write(&path, "old").await.unwrap();
        symlink(&path, backup_directory.join("source.txt.bck")).unwrap();
        let config = BackupConfig {
            kind: BackupKind::Copy,
            directories: vec![backup_directory.clone()],
            extension: "bck".into(),
        };
        let cwd = tokio::fs::canonicalize(".").await.unwrap();

        let backup = Backup::create(&path, Path::new("source.txt"), false, &cwd, &config)
            .await
            .unwrap();

        assert_eq!(backup.path.file_name().unwrap(), "source.txt.1.bck");
        assert!(
            tokio::fs::symlink_metadata(backup_directory.join("source.txt.bck"))
                .await
                .unwrap()
                .is_symlink()
        );
        tokio::fs::remove_file(backup.path).await.unwrap();
    }
}
