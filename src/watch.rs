//! Live watch: ffmpeg taps frames from a camera (or a demo stream) into one
//! atomically-rewritten JPEG; we recognize each new frame in-process and
//! print transitions. Models load once, so per-frame cost is milliseconds.

use crate::face::{face_chip, load_image, Detector, Embedder};
use crate::store;
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

pub struct WatchOpts {
    pub demo: bool,
    pub camera: String,
    pub fps: u32,
    pub duration: u32, // seconds; 0 = until ffmpeg exits / ctrl-c
    pub threshold: f32,
    pub min_face: u32,
}

fn tap_path() -> std::path::PathBuf {
    std::env::temp_dir().join("valkey-faces-tap.jpg")
}

fn spawn_ffmpeg(o: &WatchOpts, demo_dir: &str) -> Result<Child> {
    let tap = tap_path();
    let _ = std::fs::remove_file(&tap);
    let mut c = Command::new("ffmpeg");
    c.args(["-y", "-hide_banner", "-loglevel", "error"]);

    if o.demo {
        // Synthetic camera: test pattern with two demo faces overlaid on
        // co-prime cycles, so the watcher sees people come and go.
        c.args(["-re", "-f", "lavfi", "-i", "testsrc2=size=640x480:rate=15"]);
        c.args(["-loop", "1", "-framerate", "5", "-i", &format!("{demo_dir}/Barack_Obama.jpg")]);
        c.args(["-loop", "1", "-framerate", "5", "-i", &format!("{demo_dir}/Rose_Leslie.jpg")]);
        c.args(["-filter_complex",
            "[1:v]scale=200:-1[a];[2:v]scale=200:-1[b];\
             [0:v][a]overlay=x=30:y=100:enable='lt(mod(t,7),4)'[v1];\
             [v1][b]overlay=x=380:y=100:enable='lt(mod(t,11),3)'[vo];\
             [vo]fps=2,scale=512:-2[out]"]);
        c.args(["-map", "[out]"]);
    } else {
        #[cfg(target_os = "macos")]
        c.args(["-f", "avfoundation", "-framerate", "30", "-i", &o.camera]);
        #[cfg(not(target_os = "macos"))]
        c.args(["-f", "v4l2", "-framerate", "30", "-i", &o.camera]);
        c.args(["-vf", &format!("fps={},scale=512:-2", o.fps)]);
    }

    // -t goes on the OUTPUT: with -loop image inputs in the graph, an
    // input-side -t never terminates the process.
    if o.duration > 0 {
        c.args(["-t", &o.duration.to_string()]);
    }
    c.args(["-q:v", "4", "-f", "image2", "-update", "1", "-atomic_writing", "1"]);
    c.arg(&tap);
    c.stdin(Stdio::null());
    c.spawn().context("spawning ffmpeg (is it installed and on PATH?)")
}

pub fn run(
    con: &mut redis::Connection,
    det: &Detector,
    emb: &Embedder,
    o: &WatchOpts,
    demo_dir: &str,
) -> Result<()> {
    let mut child = spawn_ffmpeg(o, demo_dir)?;
    let tap = tap_path();
    println!(
        "watching {} (threshold {:.2}); ctrl-c to stop",
        if o.demo { "demo stream".into() } else { format!("camera {}", o.camera) },
        o.threshold
    );

    let mut last_mtime: Option<SystemTime> = None;
    let mut present: BTreeSet<String> = BTreeSet::new();
    let mut frames = 0u64;

    loop {
        if let Ok(Some(status)) = child.try_wait() {
            println!("stream ended ({status}); {frames} frames analyzed");
            return Ok(());
        }
        let mtime = match std::fs::metadata(&tap).and_then(|m| m.modified()) {
            Ok(m) => m,
            Err(_) => {
                std::thread::sleep(Duration::from_millis(200));
                continue;
            }
        };
        if last_mtime == Some(mtime) {
            std::thread::sleep(Duration::from_millis(1000 / (o.fps.max(1) as u64 * 2) as u64));
            continue;
        }
        last_mtime = Some(mtime);
        frames += 1;

        let t0 = Instant::now();
        let img = match load_image(tap.to_str().unwrap()) {
            Ok(i) => i,
            Err(_) => continue, // partial write race is prevented by atomic_writing, but be safe
        };
        let faces = det.detect(&img, o.min_face);
        let mut now: BTreeSet<String> = BTreeSet::new();
        for f in &faces {
            let chip = face_chip(&img, f);
            let e = emb.embed(&chip)?;
            if let Some(hit) = store::knn_search(con, &e, 1)?.into_iter().next() {
                let sim = 1.0 - hit.dist;
                if sim >= o.threshold {
                    now.insert(hit.name);
                }
            }
        }
        let ms = t0.elapsed().as_millis();

        for name in now.difference(&present) {
            println!("+ {name} entered");
        }
        for name in present.difference(&now) {
            println!("- {name} left");
        }
        if frames % 10 == 1 || now != present {
            let names: Vec<_> = now.iter().cloned().collect();
            println!(
                "  frame {frames}: {} face(s), {} known [{}] ({ms} ms)",
                faces.len(),
                now.len(),
                names.join(", ")
            );
        }
        present = now;
    }
}
