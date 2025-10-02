use std::path::Path;

use json_comments::StripComments;
use serde::de::{DeserializeOwned, Error};

pub fn parse_json_file<T: DeserializeOwned>(path: impl AsRef<Path>) -> serde_json::Result<T> {
    let path = path.as_ref();


    assert!(
        matches!(path.extension().map(|x| x.to_str()).flatten(), None | Some("json") | Some("jsonc")),
        "The given path should not have an extension, or it should be json or jsonc"
    );

    let mut path = path.with_extension("jsonc");
    if !path.exists() {
        path = path.with_extension("json");
    }

    serde_json::from_reader(
        StripComments::new(
            std::fs::File::open(path).map_err(serde_json::Error::custom)?
        )
    )
}