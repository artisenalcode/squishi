//! Honest probe: does the official Rust `magika` crate load fast (unlike
//! Kompress's 7.4s) and classify real content correctly? No assumptions —
//! measure both before considering it for content_detect.

use magika::Session;
use std::time::Instant;

fn classify(session: &mut Session, label: &str, content: &[u8]) {
    match session.identify_content_sync(content) {
        Ok(result) => println!("{label}: {}", result.info().label),
        Err(e) => println!("{label}: ERROR {e}"),
    }
}

fn main() -> magika::Result<()> {
    let t0 = Instant::now();
    let mut session = Session::new()?;
    println!("load took {:?}\n", t0.elapsed());

    let samples: Vec<(&str, &[u8])> = vec![
        ("plain prose", b"This is just a normal English sentence with no special structure at all."),
        ("json", br#"{"name": "test", "values": [1, 2, 3], "nested": {"a": true}}"#),
        (
            "log",
            b"2026-08-06 12:00:01 ERROR Connection refused at 10.0.0.4:5432\n2026-08-06 12:00:02 INFO Retrying...",
        ),
        (
            "diff",
            b"diff --git a/src/main.rs b/src/main.rs\nindex abc123..def456 100644\n--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1,3 +1,3 @@\n-let x = 5;\n+let x = 10;",
        ),
        (
            "html",
            b"<html><head><title>Test</title></head><body><h1>Hello</h1><p>World</p></body></html>",
        ),
        (
            "rust source",
            b"fn main() {\n    let x = 5;\n    println!(\"{}\", x);\n}\n",
        ),
        (
            "csv",
            b"name,age,city\nAlice,30,NYC\nBob,25,LA\nCarol,35,SF\n",
        ),
        (
            "markdown",
            b"# Title\n\nSome **bold** text and a [link](http://example.com).\n\n- item 1\n- item 2\n",
        ),
    ];

    let t1 = Instant::now();
    for (label, content) in &samples {
        classify(&mut session, label, content);
    }
    println!(
        "\n{} classifications took {:?}",
        samples.len(),
        t1.elapsed()
    );

    Ok(())
}
