// Shared Valkey helpers: pack/unpack f32 vectors LE, FT.CREATE, HSET, FT.SEARCH KNN.
use anyhow::{anyhow, Result};
use crate::face::EMBED_DIM;
use redis::{Commands, Value};

pub fn connect(url: &str) -> Result<redis::Connection> {
    let client = redis::Client::open(url)?;
    let con = client.get_connection()?;
    Ok(con)
}

/// Pack a Vec<f32> as little-endian f32 bytes (Valkey VECTOR HNSW expects this).
pub fn pack_f32le(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

pub fn unpack_f32le(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

const INDEX: &str = "faces";
const PREFIX: &str = "face:";

/// Drop (ignore errors) then create the HNSW cosine index.
pub fn create_index(con: &mut redis::Connection) -> Result<()> {
    let _: () = redis::cmd("FT.DROPINDEX")
        .arg(INDEX)
        .query(con)
        .unwrap_or(());
    let _: () = redis::cmd("FT.CREATE")
        .arg(INDEX)
        .arg("ON")
        .arg("HASH")
        .arg("PREFIX")
        .arg(1)
        .arg(PREFIX)
        .arg("SCHEMA")
        .arg("v")
        .arg("VECTOR")
        .arg("HNSW")
        .arg(6)
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(EMBED_DIM as u32)
        .arg("DISTANCE_METRIC")
        .arg("COSINE")
        .query(con)?;
    Ok(())
}

/// Slugify a name for use in keys: lowercase, non-alnum -> '-'.
pub fn slug(name: &str) -> String {
    let s: String = name
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    s.trim_matches('-').to_string()
}

/// Enroll one embedding under face:<slug>:<n>. Finds the next free n for that slug.
/// Returns the key used.
pub fn enroll(con: &mut redis::Connection, name: &str, e: &[f32]) -> Result<String> {
    let s = slug(name);
    let mut n = 0;
    loop {
        let key = format!("{}{}:{}", PREFIX, s, n);
        let exists: bool = con.exists(&key)?;
        if !exists {
            let packed = pack_f32le(e);
            let name_bytes = name.as_bytes();
            let _: () = con.hset_multiple(&key, &[("v", packed.as_slice()), ("name", name_bytes)])?;
            return Ok(key);
        }
        n += 1;
    }
}

pub struct KnnHit {
    pub name: String,
    pub dist: f32,
}

/// KNN search: query vector -> nearest k hits (name + cosine distance).
pub fn knn_search(con: &mut redis::Connection, query: &[f32], k: usize) -> Result<Vec<KnnHit>> {
    let packed = pack_f32le(query);
    let search = format!("*=>[KNN {} @v $q AS dist]", k);
    let v: Value = redis::cmd("FT.SEARCH")
        .arg(INDEX)
        .arg(&search)
        .arg("PARAMS")
        .arg(2)
        .arg("q")
        .arg(packed.as_slice())
        .arg("RETURN")
        .arg(2)
        .arg("name")
        .arg("dist")
        .arg("DIALECT")
        .arg(2)
        .query(con)?;
    parse_search(v)
}

// Parse a FT.SEARCH DIALECT 2 reply.
// Shape (redis crate Value): array of [count, key1, [field, value, ...], key2, [...], ...]
fn parse_search(v: Value) -> Result<Vec<KnnHit>> {
    let arr = match v {
        Value::Array(a) => a,
        other => return Err(anyhow!("unexpected FT.SEARCH reply: {:?}", other)),
    };
    if arr.is_empty() {
        return Ok(vec![]);
    }
    let mut hits = Vec::new();
    let mut i = 1; // skip count at index 0
    while i + 1 < arr.len() {
        // arr[i] = key, arr[i+1] = fields array
        let fields = match &arr[i + 1] {
            Value::Array(f) => f,
            other => return Err(anyhow!("unexpected fields reply: {:?}", other)),
        };
        let mut name = String::new();
        let mut dist = f32::NAN;
        let mut j = 0;
        while j + 1 < fields.len() {
            let fname = as_string(&fields[j]);
            match fname.as_str() {
                "name" => name = as_string(&fields[j + 1]),
                "dist" => {
                    dist = match &fields[j + 1] {
                        Value::Double(d) => *d as f32,
                        Value::Int(n) => *n as f32,
                        Value::BulkString(b) => {
                            String::from_utf8_lossy(b).parse().unwrap_or(f32::NAN)
                        }
                        _ => f32::NAN,
                    }
                }
                _ => {}
            }
            j += 2;
        }
        if !name.is_empty() {
            hits.push(KnnHit { name, dist });
        }
        i += 2;
    }
    Ok(hits)
}

fn as_string(v: &Value) -> String {
    match v {
        Value::BulkString(b) => String::from_utf8_lossy(b).to_string(),
        Value::SimpleString(s) => s.clone(),
        Value::Int(n) => n.to_string(),
        Value::Double(d) => d.to_string(),
        _ => String::new(),
    }
}

/// List enrolled people: name -> number of entries.
pub fn list(con: &mut redis::Connection) -> Result<Vec<(String, usize)>> {
    use std::collections::BTreeMap;
    let keys: Vec<String> = redis::cmd("KEYS").arg(format!("{}*", PREFIX)).query(con)?;
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for k in keys {
        let name: Option<String> = con.hget(&k, "name")?;
        if let Some(n) = name {
            *counts.entry(n).or_default() += 1;
        }
    }
    Ok(counts.into_iter().collect())
}

/// Remove every entry for a person. Returns how many keys were deleted.
pub fn forget(con: &mut redis::Connection, name: &str) -> Result<usize> {
    let s = slug(name);
    let keys: Vec<String> = redis::cmd("KEYS").arg(format!("{}{}:*", PREFIX, s)).query(con)?;
    let n = keys.len();
    for k in &keys {
        let _: () = con.del(k)?;
    }
    Ok(n)
}

/// True if the index already exists.
pub fn index_exists(con: &mut redis::Connection) -> bool {
    redis::cmd("FT.INFO").arg(INDEX).query::<Value>(con).is_ok()
}
