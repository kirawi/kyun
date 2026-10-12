use std::io::{self, Write};
use std::sync::Arc;

use bytes::Bytes;
use futures_util::{StreamExt, stream};
use reqwest::Body;
use tokio::sync::{
    Mutex, OwnedMutexGuard,
    mpsc::{Sender, channel},
};

const BUFFER_SIZE: usize = 8192;
type Buffer = [u8; BUFFER_SIZE];

struct StreamWriter {
    buf: Arc<Mutex<Buffer>>,
    written: usize,
    guard: Option<OwnedMutexGuard<Buffer>>,
    tx: Sender<Result<Bytes, io::Error>>,
}

impl io::Write for StreamWriter {
    fn write(&mut self, src: &[u8]) -> io::Result<usize> {
        if self.guard.is_none() {
            // Wait for `Body::wrap_stream` to release the buffer.
            self.guard = Some(self.buf.clone().blocking_lock_owned());
        }

        let guard = self.guard.as_mut().unwrap();
        let n = src.len().min(BUFFER_SIZE - self.written);
        guard[self.written..self.written + n].copy_from_slice(&src[..n]);
        self.written += n;

        if self.written == BUFFER_SIZE {
            self.publish()?;
        }

        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.written != 0 {
            self.publish()?;
        }
        Ok(())
    }
}

impl StreamWriter {
    fn publish(&mut self) -> io::Result<()> {
        debug_assert_ne!(self.written, 0);

        let guard = self.guard.take().unwrap();
        let len = std::mem::replace(&mut self.written, 0);
        let chunk = BorrowedChunk { guard, len };
        let bytes = Bytes::from_owner(chunk);

        self.tx
            .blocking_send(Ok(bytes))
            .map_err(|_| io::Error::other("request body receiver disappeared"))
    }
}

struct BorrowedChunk {
    guard: OwnedMutexGuard<Buffer>,
    len: usize,
}

impl AsRef<[u8]> for BorrowedChunk {
    fn as_ref(&self) -> &[u8] {
        &self.guard[..self.len]
    }
}

/// Uses a constant-sized buffer to transmit data, reducing peak memory usage for very large data where the data
/// may otherwise be allocated to an unbounded `Vec<u8>`.
pub fn serialize_to_json<T>(data: T) -> reqwest::Body
where
    T: serde::Serialize + Send + 'static,
{
    let buf = Arc::new(Mutex::new([0_u8; BUFFER_SIZE]));
    let (tx, rx) = channel::<Result<Bytes, io::Error>>(1);
    let stream = stream::once(async move {
        tokio::task::spawn_blocking(move || {
            let mut writer = StreamWriter {
                buf,
                written: 0,
                guard: None,
                tx,
            };
            if let Err(error) = serde_json::to_writer(&mut writer, &data) {
                let _ = writer.tx.blocking_send(Err(io::Error::other(error)));
            } else {
                let _ = writer.flush();
            }
        });
        tokio_stream::wrappers::ReceiverStream::new(rx)
    })
    .flatten();
    Body::wrap_stream(stream)
}
