use std::{
    collections::VecDeque,
    env,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{
        LazyLock, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use burn::tensor::ops::FloatTensor;
use crossbeam::utils::Backoff;
use parking_lot::Mutex;
use rusqlite::{Connection, ToSql};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::TrackingBackend;

#[derive(PartialEq, Eq)]
struct TrackedTensor {
    from_hash: [u8; 32],
    to_hash: Option<[u8; 32]>,
    operation: &'static str,
    extra_data: String,
    time: u64,
}

impl PartialOrd for TrackedTensor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TrackedTensor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.time.cmp(&other.time)
    }
}

static ARTIFACT_DIR: OnceLock<PathBuf> = OnceLock::new();
static TRACKED_TENSORS: LazyLock<Mutex<Vec<TrackedTensor>>> = LazyLock::new(init_task);
static TRACKED_COUNT: AtomicUsize = AtomicUsize::new(0);
static START_TIME: LazyLock<Instant> = LazyLock::new(|| Instant::now());

fn get_time() -> u64 {
    (START_TIME.elapsed().as_nanos() % u64::MAX as u128) as u64
}

pub fn set_artifact_dir(path: PathBuf) {
    let _ = ARTIFACT_DIR.set(path);
}

pub(crate) fn hash_tensor(tensor: &FloatTensor<TrackingBackend>) -> [u8; 32] {
    let mut digest = Sha256::new();
    let data = tensor
        .primitive
        .client
        .read_one(tensor.primitive.handle.clone());
    for dim in tensor.primitive.shape.as_slice() {
        digest.update(dim.to_ne_bytes());
    }
    digest.update(&*data);
    let hash = digest.finalize();
    hash.try_into().unwrap()
}

pub(crate) struct TrackedTensorBuilder {
    from_hash: [u8; 32],
    operation: &'static str,
    extra_data: String,
}

pub(crate) fn start_tracking_tensor(
    tensor: &FloatTensor<TrackingBackend>,
    operation: &'static str,
    extra_data: Value,
) -> TrackedTensorBuilder {
    TrackedTensorBuilder {
        from_hash: hash_tensor(tensor),
        operation,
        extra_data: extra_data.to_string(),
    }
}

// pub(crate) fn start_tracking_tensor_raw(hash: [u8; 32], operation: &'static str, extra_data: Value) -> TrackedTensorBuilder {
//     TrackedTensorBuilder {
//         from_hash: hash,
//         operation,
//         extra_data: extra_data.to_string(),
//     }
// }

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
pub(crate) fn track_tensor(
    tensor: &FloatTensor<TrackingBackend>,
    operation: &'static str,
    extra_data: Value,
) {
    start_tracking_tensor(tensor, operation, extra_data).finish_no_tensor();
}

impl TrackedTensorBuilder {
    pub(crate) fn finish_ref(self, tensor: &FloatTensor<TrackingBackend>) {
        TRACKED_TENSORS.lock().push(TrackedTensor {
            from_hash: self.from_hash,
            to_hash: Some(hash_tensor(tensor)),
            operation: self.operation,
            extra_data: self.extra_data,
            time: get_time(),
        });
    }

    // pub(crate) fn finish_raw(self, hash: [u8; 32]) {
    //     TRACKED_TENSORS.lock().push(
    //         TrackedTensor { from_hash: self.from_hash, to_hash: Some(hash), operation: self.operation, extra_data: self.extra_data, time: get_time() }
    //     );
    // }

    pub(crate) fn finish(
        self,
        tensor: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        self.finish_ref(&tensor);
        tensor
    }

    pub(crate) fn finish_no_tensor(self) {
        TRACKED_TENSORS.lock().push(TrackedTensor {
            from_hash: self.from_hash,
            to_hash: None,
            operation: self.operation,
            extra_data: self.extra_data,
            time: get_time(),
        });
    }
}

pub(crate) fn finish_iter(
    builders: impl IntoIterator<Item = TrackedTensorBuilder>,
    to_hash: Option<[u8; 32]>,
) {
    let mut guard = TRACKED_TENSORS.lock();
    let time = get_time();
    guard.extend(builders.into_iter().map(|builder| TrackedTensor {
        from_hash: builder.from_hash,
        to_hash,
        operation: builder.operation,
        extra_data: builder.extra_data,
        time,
    }));
}

fn init_task() -> Mutex<Vec<TrackedTensor>> {
    let batch_size: NonZeroUsize = env::var("BATCH_SIZE")
        .unwrap_or_else(|_| String::from("32"))
        .parse()
        .expect("Expected BATCH_SIZE to be a positive integer");
    let workers: NonZeroUsize = env::var("WORKERS")
        .unwrap_or_else(|_| String::from("1"))
        .parse()
        .expect("Expected WORKERS to be a positive integer");

    let thread = move || {
        let artifact_dir = ARTIFACT_DIR
            .get()
            .map(|x| x.as_path())
            .unwrap_or(Path::new("."));
        let conn = Connection::open(artifact_dir.join("tracked-tensors.sqlite"))
            .expect("Expected SQLite database to be creatable/openable");

        conn.execute(
            "CREATE TABLE IF NOT EXISTS tensors (
                row_id INTEGER PRIMARY KEY,
                from_hash TEXT NOT NULL,
                to_hash TEXT,
                operation TEXT NOT NULL,
                extra_data TEXT NOT NULL
            ) STRICT;",
            (),
        )
        .unwrap();
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

        let mut buffer = VecDeque::with_capacity(batch_size.get());
        let mut batch_sql = String::from(
            "INSERT INTO tensors
                (from_hash, to_hash, operation, extra_data)
            VALUES
                (HEX(?), HEX(?), ?, ?)",
        );
        for _ in 0..batch_size.get() - 1 {
            batch_sql.push_str(",(HEX(?), HEX(?), ?, ?)");
        }

        let mut batch_insert = conn.prepare(&batch_sql).unwrap();
        let backoff = Backoff::new();

        loop {
            {
                let mut guard = TRACKED_TENSORS.lock();
                let len = guard.len();
                if len == 0 {
                    drop(guard);
                    backoff.snooze();
                    continue;
                }
                backoff.reset();
                buffer.extend(guard.drain(..));
                TRACKED_COUNT.fetch_add(len, Ordering::Relaxed);
            }
            while buffer.len() >= batch_size.get() {
                let mut params: Vec<&dyn ToSql> = Vec::with_capacity(batch_size.get());
                params.extend(buffer.iter().take(batch_size.get()).flat_map(
                    |tensor: &TrackedTensor| {
                        [
                            &tensor.from_hash as &dyn ToSql,
                            &tensor.to_hash,
                            &tensor.operation,
                            &tensor.extra_data,
                        ]
                    },
                ));
                batch_insert
                    .execute(&*params)
                    .expect("Expected to insert tracked tensors");
                buffer.drain(0..batch_size.get());
            }
        }

        // let mut single_insert = conn.prepare(
        //     "INSERT INTO tensors
        //         (tensor_hash, operation, extra_data)
        //     VALUES
        //         (HEX(?), HEX(?), ?, ?)
        //     "
        // ).unwrap();
        // for tensor in buffer {
        //     single_insert.execute(
        //         params![tensor.from_hash, tensor.to_hash, tensor.operation, tensor.extra_data]
        //     ).expect("Expected to insert tensor");
        // }
    };

    for _ in 0..workers.get() {
        std::thread::spawn(thread.clone());
    }

    Mutex::new(Vec::new())
}
