use std::{
    collections::VecDeque,
    fs::File,
    hash::Hash,
    io::{BufWriter, Write},
    path::Path,
    rc::Rc,
    sync::atomic::{AtomicBool, Ordering},
};

use rusqlite::{Connection, params};
use rustc_hash::FxHashSet;

#[derive(Clone, Debug)]
struct Visitor {
    start_hash: Rc<str>,
    operations: Vec<Rc<str>>,
    last_hash: Rc<str>,
}

#[derive(Eq)]
struct UniquePipeline {
    start_hash: Rc<str>,
    operations: Vec<Rc<str>>,
}

impl PartialEq for UniquePipeline {
    fn eq(&self, other: &Self) -> bool {
        self.operations
            .iter()
            .zip(other.operations.iter())
            .all(|(a, b)| a.as_ptr() == b.as_ptr())
    }
}

impl Hash for UniquePipeline {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.operations.hash(state);
    }
}

fn main() {
    let conn = Connection::open(
        std::env::args()
            .nth(1)
            .expect("Expected SQLite Database path to be provided"),
    )
    .expect("Expected SQLite database to be valid");
    let mut stmt = conn
        .prepare("SELECT DISTINCT to_hash from tensors WHERE operation = 'float_cat'")
        .unwrap();
    let mut string_interner = FxHashSet::<Rc<str>>::default();

    let mut requirements = vec![];
    let tmp;
    if Path::new("tensor_requirements.txt").exists() {
        tmp = std::fs::read_to_string("tensor_requirements.txt").unwrap_or_default();
        requirements = tmp
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect();
    }

    let mut queue = VecDeque::new();
    let mut seen = FxHashSet::default();
    {
        let mut rows = stmt.query(()).unwrap();
        while let Some(float_cat_row) = rows.next().unwrap() {
            let to_hash: Rc<str> = float_cat_row.get_unwrap("to_hash");
            seen.insert(to_hash.clone());
            queue.push_back(Visitor {
                start_hash: to_hash.clone().into(),
                operations: vec![],
                last_hash: to_hash,
            });
        }
    }
    stmt = conn
        .prepare("SELECT DISTINCT to_hash, operation from tensors WHERE from_hash = ?")
        .unwrap();
    let mut done = FxHashSet::default();
    let mut output = BufWriter::new(
        File::create("tensor_graph.txt").expect("Expected tensor_graph.txt to be created"),
    );
    writeln!(output, "digraph G {{").unwrap();

    let ctrlc_pressed: &_ = Box::leak(Box::new(AtomicBool::new(false)));

    ctrlc::set_handler(move || {
        ctrlc_pressed.store(true, Ordering::Relaxed);
        println!("Cancelling due to Ctrl-C ...");
    })
    .expect("Error setting Ctrl-C handler");

    while let Some(visitor) = queue.pop_front() {
        let mut rows = stmt.query(params![visitor.last_hash]).unwrap();
        let mut empty = true;
        while let Some(row) = rows.next().unwrap() {
            let to_hash: Rc<str> = row.get_unwrap("to_hash");
            if !seen.insert(to_hash.clone()) {
                continue;
            }
            empty = false;
            let operation: Rc<str> = row.get_unwrap("operation");
            let operation = if let Some(op) = string_interner.get(&operation) {
                op.clone()
            } else {
                string_interner.insert(operation.clone());
                string_interner.get(&operation).unwrap().clone()
            };
            let mut visitor = visitor.clone();
            visitor.operations.push(operation);
            visitor.last_hash = to_hash;
            queue.push_back(visitor);
        }
        if empty {
            if visitor.operations.is_empty() {
                continue;
            }
            if !requirements.is_empty() {
                let mut challenge = VecDeque::from_iter(requirements.iter().copied());
                for op in &visitor.operations {
                    if **challenge.front().unwrap() == **op {
                        challenge.pop_front();
                        if challenge.is_empty() {
                            break;
                        }
                    }
                }
                if !challenge.is_empty() {
                    continue;
                }
            }
            let pipeline = UniquePipeline {
                start_hash: visitor.start_hash,
                operations: visitor.operations,
            };
            if !done.contains(&pipeline) {
                let i = done.len();
                writeln!(output, "    subgraph cluster_{i} {{").unwrap();
                let mut hash = vec![];
                for chunk in pipeline.start_hash.as_bytes().chunks(16) {
                    if !hash.is_empty() {
                        hash.push(b'\\');
                        hash.push(b'n');
                    }
                    hash.extend_from_slice(chunk);
                }
                writeln!(
                    output,
                    "        label = \"{}\";",
                    String::from_utf8(hash).unwrap()
                )
                .unwrap();
                for (j, op) in pipeline.operations.iter().enumerate() {
                    writeln!(output, "        n_{i}_{j}[label=\"{op}\"];").unwrap();
                }
                write!(output, "        n_{i}_0").unwrap();
                for j in 1..pipeline.operations.len() {
                    write!(output, " -> n_{i}_{j}").unwrap();
                }
                writeln!(output, ";\n    }}").unwrap();
                output.flush().unwrap();
                done.insert(pipeline);
            }
        }
        if ctrlc_pressed.load(Ordering::Relaxed) {
            break;
        }
    }
    writeln!(output, "}}").unwrap();
    output.flush().unwrap();
}
