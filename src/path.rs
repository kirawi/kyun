use std::{path::PathBuf, sync::OnceLock};

static CWD: OnceLock<PathBuf> = OnceLock::new();

/// Gets the process working directory while preserving its logical `PWD` spelling when valid.
pub fn current_working_dir() -> PathBuf {
    CWD.get_or_init(|| {
        // A deleted working directory degrades to a path tools cannot resolve, rather than panicking
        // mid-session inside this lazily initialized lock.
        let mut cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let pwd = std::env::var_os("PWD");

        #[cfg(windows)]
        let pwd = pwd.or_else(|| std::env::var_os("CD"));

        if let Some(pwd) = pwd.map(PathBuf::from)
            && pwd.canonicalize().ok().as_ref() == Some(&cwd)
        {
            cwd = pwd;
        }
        cwd
    })
    .clone()
}

pub fn working_directory_root() -> PathBuf {
    let cwd = current_working_dir();
    cwd.canonicalize().unwrap_or(cwd)
}
