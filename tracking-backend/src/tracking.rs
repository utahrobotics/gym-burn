use std::{env, num::NonZeroUsize, path::{Path, PathBuf}, sync::{LazyLock, OnceLock, atomic::{AtomicUsize, Ordering}}, time::Duration};

use burn::tensor::ops::FloatTensor;
use crossbeam::channel::Sender;
use rusqlite::{Connection, ToSql, params};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::TrackingBackend;

struct TrackedTensor {
    from_hash: [u8; 32],
    to_hash: Option<[u8; 32]>,
    operation: &'static str,
    extra_data: String
}


static ARTIFACT_DIR: OnceLock<PathBuf> = OnceLock::new();
static TRACKED_TENSOR_SENDER: LazyLock<Sender<TrackedTensor>> = LazyLock::new(init_task);
static TRACKED_COUNT: AtomicUsize = AtomicUsize::new(0);


pub fn set_artifact_dir(path: PathBuf) {
    let _ = ARTIFACT_DIR.set(path);
}


pub(crate) fn hash_tensor(tensor: &FloatTensor<TrackingBackend>) -> [u8; 32] {
    let mut digest = Sha256::new();
    let data = tensor.client.read_one(tensor.handle.clone());
    for dim in tensor.shape.as_slice() {
        digest.update(dim.to_ne_bytes());
    }
    digest.update(&*data);
    let hash = digest.finalize();
    hash.try_into().unwrap()
}


pub(crate) struct TrackedTensorBuilder {
    from_hash: [u8; 32],
    operation: &'static str,
    extra_data: String
}


pub(crate) fn start_tracking_tensor(tensor: &FloatTensor<TrackingBackend>, operation: &'static str, extra_data: Value) -> TrackedTensorBuilder {
    TrackedTensorBuilder {
        from_hash: hash_tensor(tensor),
        operation,
        extra_data: extra_data.to_string(),
    }
}


pub(crate) fn start_tracking_tensor_raw(hash: [u8; 32], operation: &'static str, extra_data: Value) -> TrackedTensorBuilder {
    TrackedTensorBuilder {
        from_hash: hash,
        operation,
        extra_data: extra_data.to_string(),
    }
}

pub fn wait_until_paused() {
    let mut init = TRACKED_COUNT.load(Ordering::Relaxed);
    loop {
        std::thread::sleep(Duration::from_secs(2));
        let new = TRACKED_COUNT.load(Ordering::Relaxed);
        if new == init {
            break;
        }
        init = new;
    }
}

#[allow(unused)]
pub(crate) fn track_tensor(tensor: &FloatTensor<TrackingBackend>, operation: &'static str, extra_data: Value) {
    start_tracking_tensor(tensor, operation, extra_data).finish_no_tensor();
}


impl TrackedTensorBuilder {
    pub(crate) fn finish_ref(self, tensor: &FloatTensor<TrackingBackend>) {
        let _ = TRACKED_TENSOR_SENDER.send(
            TrackedTensor { from_hash: self.from_hash, to_hash: Some(hash_tensor(tensor)), operation: self.operation, extra_data: self.extra_data }
        );
    }
    pub(crate) fn finish_raw(self, hash: [u8; 32]) {
        let _ = TRACKED_TENSOR_SENDER.send(
            TrackedTensor { from_hash: self.from_hash, to_hash: Some(hash), operation: self.operation, extra_data: self.extra_data }
        );
    }
    pub(crate) fn finish(self, tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        self.finish_ref(&tensor);
        tensor
    }
    pub(crate) fn finish_no_tensor(self) {
        let _ = TRACKED_TENSOR_SENDER.send(
            TrackedTensor { from_hash: self.from_hash, to_hash: None, operation: self.operation, extra_data: self.extra_data }
        );
    }
}


fn init_task() -> Sender<TrackedTensor> {
    let batch_size: NonZeroUsize = env::var("BATCH_SIZE").unwrap_or_else(|_| String::from("32")).parse().expect("Expected BATCH_SIZE to be a positive integer");
    let workers: NonZeroUsize = env::var("WORKERS").unwrap_or_else(|_| String::from("1")).parse().expect("Expected WORKERS to be a positive integer");

    let (sender, receiver) = crossbeam::channel::bounded(batch_size.get() * 32 * workers.get());

    let thread = move || {
        let artifact_dir = ARTIFACT_DIR.get().map(|x| x.as_path()).unwrap_or(Path::new("."));
        let conn = Connection::open(artifact_dir.join("tracked-tensors.sqlite")).expect("Expected SQLite database to be creatable/openable");
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tensors (
                row_id INTEGER PRIMARY KEY,
                from_hash TEXT NOT NULL,
                to_hash TEXT,
                operation TEXT NOT NULL,
                extra_data TEXT NOT NULL
            ) STRICT;",
            ()
        ).unwrap();
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_from_hash ON tensors (from_hash);",
            (),
        )
        .unwrap();
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_to_hash ON tensors (to_hash);",
            (),
        )
        .unwrap();
        
        let mut buffer = Vec::with_capacity(batch_size.get());
        let mut batch_sql = String::from(
            "INSERT INTO tensors
                (from_hash, to_hash, operation, extra_data)
            VALUES
                (HEX(?), HEX(?), ?, ?)"
        );
        for _ in 0..batch_size.get() - 1 {
            batch_sql.push_str(",(HEX(?), HEX(?), ?, ?)");
        }

        let mut batch_insert = conn.prepare(&batch_sql).unwrap();

        while let Ok(tensor) = receiver.recv() {
            buffer.push(tensor);
            TRACKED_COUNT.fetch_add(1, Ordering::Relaxed);
            if buffer.len() < batch_size.get() {
                continue;
            }
            let mut params: Vec<&dyn ToSql> = Vec::with_capacity(batch_size.get());
            params.extend(
                buffer.iter()
                    .flat_map(|tensor: &TrackedTensor| {
                        [
                            &tensor.from_hash as &dyn ToSql,
                            &tensor.to_hash as &dyn ToSql,
                            &tensor.operation,
                            &tensor.extra_data
                        ]
                    })
            );
            batch_insert.execute(
                &*params
            ).expect("Expected to insert tracked tensors");
            buffer.clear();
        }

        let mut single_insert = conn.prepare(
            "INSERT INTO tensors
                (tensor_hash, operation, extra_data)
            VALUES
                (HEX(?), HEX(?), ?, ?)
            "
        ).unwrap();
        for tensor in buffer {
            single_insert.execute(
                params![tensor.from_hash, tensor.to_hash, tensor.operation, tensor.extra_data]
            ).expect("Expected to insert tensor");
        }
    };

    for _ in 0..workers.get() {
        std::thread::spawn(thread.clone());
    }

    sender
}