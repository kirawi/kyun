use std::io;

use anyhow::{Result, anyhow};
use encoding_rs::{self as encoding, Encoding};
use ropey::{Rope, RopeBuilder};

const BUF_SIZE: usize = 8192;

/// A cheap heuristic to guess whether a file may be a binary file.
pub fn is_binary(buffer: &[u8]) -> bool {
    const BYTE_ORDER_MARKS: &[&[u8]] = &[
        &[0xEF, 0xBB, 0xBF], // UTF-8
        &[0xFE, 0xFF],       // UTF-16BE
        &[0xFF, 0xFE],       // UTF-16LE
    ];

    if BYTE_ORDER_MARKS.iter().any(|bom| buffer.starts_with(bom)) {
        return false;
    }

    let scan = &buffer[..buffer.len().min(1024)];
    scan.contains(&0) || buffer.starts_with(b"%PDF") || buffer.starts_with(b"\x89PNG")
}

// TODO: could be made async; sync not used/needed
/// Decodes a stream of bytes into UTF-8, returning a `Rope`, the encoding it was decoded as, and its BOM.
/// The optional `encoding` parameter can be used to override encoding auto-detection.
pub fn from_reader<R: std::io::Read + ?Sized>(
    reader: &mut R,
    encoding: Option<&'static Encoding>,
) -> Result<(Rope, &'static Encoding, bool), io::Error> {
    let mut buf = [0u8; BUF_SIZE];
    let mut buf_out = [0u8; BUF_SIZE];
    let mut builder = RopeBuilder::new();

    let (encoding, has_bom, mut decoder, read) =
        read_and_detect_encoding(reader, encoding, &mut buf)?;

    let mut slice = &buf[..read];
    let mut is_empty = read == 0;

    // SAFETY: `buf_out` is a zero-initialized array, thus it will always
    // contain valid UTF-8.
    let buf_str = unsafe { std::str::from_utf8_unchecked_mut(&mut buf_out[..]) };
    let mut total_written = 0usize;
    loop {
        let mut total_read = 0usize;

        // Handle cases where the output buffer fills before the input is fully decoded.
        loop {
            let (result, read, written) = decoder.decode_to_str_without_replacement(
                &slice[total_read..],
                &mut buf_str[total_written..],
                is_empty,
            );

            total_read += read;
            total_written += written;
            match result {
                encoding::DecoderResult::InputEmpty => {
                    debug_assert_eq!(slice.len(), total_read);
                    break;
                }
                encoding::DecoderResult::OutputFull => {
                    debug_assert!(slice.len() > total_read);
                    builder.append(&buf_str[..total_written]);
                    total_written = 0;
                }
                encoding::DecoderResult::Malformed(_, _) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "input contains malformed text for its detected encoding",
                    ));
                }
            }
        }
        if is_empty {
            debug_assert_eq!(reader.read(&mut buf)?, 0);
            builder.append(&buf_str[..total_written]);
            break;
        }

        let read = reader.read(&mut buf)?;
        slice = &buf[..read];
        is_empty = read == 0;
    }
    let rope = builder.finish();
    Ok((rope, encoding, has_bom))
}

/// Determines the encoding to decode the reader's contents with: an explicit override if given, otherwise a BOM,
/// otherwise a `chardetng` guess. As detection requires sample data, the first bufferful is read into `buf` and
/// its length returned so the caller's loop decodes those same bytes.
fn read_and_detect_encoding<R: std::io::Read + ?Sized>(
    reader: &mut R,
    encoding: Option<&'static Encoding>,
    buf: &mut [u8],
) -> Result<(&'static Encoding, bool, encoding::Decoder, usize), io::Error> {
    let read = reader.read(buf)?;
    let is_empty = read == 0;
    let (encoding, has_bom) = encoding
        .map(|encoding| (encoding, false))
        .or_else(|| encoding::Encoding::for_bom(buf).map(|(encoding, _bom_size)| (encoding, true)))
        .unwrap_or_else(|| {
            let mut encoding_detector =
                chardetng::EncodingDetector::new(chardetng::Iso2022JpDetection::Allow);
            // Feed only the bytes actually read: the NUL padding beyond them would skew detection, and an
            // empty read is the final (and only) sample.
            encoding_detector.feed(&buf[..read], is_empty);
            (
                encoding_detector.guess(None, chardetng::Utf8Detection::Allow),
                false,
            )
        });
    let decoder = encoding.new_decoder();

    Ok((encoding, has_bom, decoder, read))
}

enum Encoder {
    Utf16Be,
    Utf16Le,
    EncodingRs(encoding::Encoder),
}

impl Encoder {
    fn from_encoding(encoding: &'static Encoding) -> Self {
        if encoding == encoding::UTF_16BE {
            Self::Utf16Be
        } else if encoding == encoding::UTF_16LE {
            Self::Utf16Le
        } else {
            Self::EncodingRs(encoding.new_encoder())
        }
    }

    fn encode_from_utf8(
        &mut self,
        src: &str,
        dst: &mut [u8],
        is_empty: bool,
    ) -> (encoding::CoderResult, usize, usize, bool) {
        // An empty chunk only reaches the encoders as the end-of-stream sentinel, where `is_empty` flushes
        // pending state (ISO-2022-JP's terminal escape); the UTF-16 arms simply write nothing.
        if src.is_empty() && !is_empty {
            return (encoding::CoderResult::InputEmpty, 0, 0, false);
        }
        let mut write_to_buf = |convert: fn(u16) -> [u8; 2]| {
            let to_write = src.char_indices().map(|(index, character)| {
                let mut encoded = [0, 0];
                (
                    index,
                    character
                        .encode_utf16(&mut encoded)
                        .iter()
                        .flat_map(|character| convert(*character))
                        .collect::<Vec<u8>>(),
                )
            });

            let mut total_written = 0usize;

            for (index, utf16_bytes) in to_write {
                let character_size = utf16_bytes.len();

                if dst.len() < total_written + character_size {
                    return (
                        encoding::CoderResult::OutputFull,
                        index,
                        total_written,
                        false,
                    );
                }

                for character in utf16_bytes {
                    dst[total_written] = character;
                    total_written += 1;
                }
            }

            (
                encoding::CoderResult::InputEmpty,
                src.len(),
                total_written,
                false,
            )
        };

        match self {
            Self::Utf16Be => write_to_buf(u16::to_be_bytes),
            Self::Utf16Le => write_to_buf(u16::to_le_bytes),
            Self::EncodingRs(encoder) => {
                let (result, read, written, had_unmappables) =
                    encoder.encode_from_utf8(src, dst, is_empty);

                (result, read, written, had_unmappables)
            }
        }
    }
}

/// The byte length [`to_writer`] would produce for `rope`, including any BOM: mutations report it as the
/// on-disk size, which differs from the rope's UTF-8 length for other encodings.
pub fn encoded_len(rope: &Rope, (encoding, has_bom): (&'static Encoding, bool)) -> usize {
    let mut encoder = Encoder::from_encoding(encoding);
    let mut buf = [0u8; BUF_SIZE];
    let mut total = if has_bom {
        apply_bom(encoding, &mut buf)
    } else {
        0
    };

    // A `Rope`'s chunks are of unknown size, so the end of iteration cannot be predicted ahead of time.
    // Empty chunks are filtered out so an empty chunk sentinel can signal the end of iteration.
    let iter = rope
        .chunks()
        .filter(|c| !c.is_empty())
        .chain(std::iter::once(""));
    for chunk in iter {
        let is_empty = chunk.is_empty();
        let mut read = 0usize;
        loop {
            let (result, consumed, written, _) =
                encoder.encode_from_utf8(&chunk[read..], &mut buf, is_empty);
            total += written;
            read += consumed;
            if matches!(result, encoding::CoderResult::InputEmpty) {
                break;
            }
        }
    }
    total
}

fn apply_bom(encoding: &'static Encoding, buf: &mut [u8; BUF_SIZE]) -> usize {
    if encoding == encoding_rs::UTF_8 {
        buf[0] = 0xef;
        buf[1] = 0xbb;
        buf[2] = 0xbf;
        3
    } else if encoding == encoding_rs::UTF_16BE {
        buf[0] = 0xfe;
        buf[1] = 0xff;
        2
    } else if encoding == encoding_rs::UTF_16LE {
        buf[0] = 0xff;
        buf[1] = 0xfe;
        2
    } else {
        0
    }
}

/// Encodes the text inside `rope` into the given `encoding` and writes the
/// encoded output into `writer.`
pub async fn to_writer<'a, W: tokio::io::AsyncWriteExt + Unpin + ?Sized>(
    writer: &'a mut W,
    encoding_with_bom_info: (&'static Encoding, bool),
    rope: &'a Rope,
) -> Result<()> {
    let (encoding, has_bom) = encoding_with_bom_info;

    // A `Rope`'s chunks are of unknown size, so the end of iteration cannot be predicted ahead of time.
    // Empty chunks are filtered out so that we can use an empty chunk sentinel to signal the end of
    // iteration when writing.
    let iter = rope
        .chunks()
        .filter(|c| !c.is_empty())
        .chain(std::iter::once(""));
    let mut buf = [0u8; BUF_SIZE];

    let mut total_written = if has_bom {
        apply_bom(encoding, &mut buf)
    } else {
        0
    };

    let mut encoder = Encoder::from_encoding(encoding);
    for chunk in iter {
        let is_empty = chunk.is_empty();
        let mut total_read = 0usize;

        // Handle cases where the output buffer fills before the input is fully encoded.
        loop {
            let (result, read, written, had_unmappables) =
                encoder.encode_from_utf8(&chunk[total_read..], &mut buf[total_written..], is_empty);

            if had_unmappables {
                return Err(anyhow!("text cannot be represented in the file's encoding"));
            }

            total_read += read;
            total_written += written;
            match result {
                encoding::CoderResult::InputEmpty => {
                    debug_assert_eq!(chunk.len(), total_read);
                    debug_assert!(buf.len() >= total_written);
                    break;
                }
                encoding::CoderResult::OutputFull => {
                    debug_assert!(chunk.len() > total_read);
                    writer.write_all(&buf[..total_written]).await?;
                    total_written = 0;
                }
            }
        }

        if is_empty {
            writer.write_all(&buf[..total_written]).await?;
            writer.flush().await?;
            break;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn encoded_len_counts_the_bom_and_the_target_encoding() {
        // UTF-8 without a BOM matches the rope's own byte length.
        let ascii = Rope::from("ab");
        assert_eq!(encoded_len(&ascii, (encoding_rs::UTF_8, false)), 2);
        // A UTF-8 BOM adds three bytes.
        assert_eq!(encoded_len(&ascii, (encoding_rs::UTF_8, true)), 5);
        // UTF-16LE doubles the bytes and adds a two-byte BOM.
        assert_eq!(encoded_len(&ascii, (encoding_rs::UTF_16LE, true)), 6);
    }

    #[tokio::test]
    async fn iso_2022_jp_output_ends_with_the_ascii_escape() {
        // ISO-2022-JP is the only encoding with pending encoder state; without the final flush the file
        // round-trips but its bytes are not canonical (FINDINGS Low).
        // Text ending in a double-byte character is the case that needs the flush: without it the file
        // round-trips but never escapes back to ASCII mode.
        let rope = Rope::from("\u{3042}");
        let mut bytes = Vec::new();
        to_writer(&mut bytes, (encoding_rs::ISO_2022_JP, false), &rope)
            .await
            .unwrap();

        assert!(bytes.ends_with(b"\x1b(B"), "{bytes:?}");
        let (text, _, _) = from_reader(&mut Cursor::new(&bytes), None).unwrap();
        assert_eq!(text.to_string(), "\u{3042}");
    }
}
