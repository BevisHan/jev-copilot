//! Jev Copilot — a web conversation co-pilot driven by Jev.
//!
//! Pipeline, mirroring jev-chat-jarvis: judge first, then write.
//!
//!   conversation ──▶ Jev: 7 judgements (one request, ~1s)
//!                          │
//!                          ▼
//!                    chat LLM: draft 3 candidate replies
//!                          │
//!                          ▼
//!                    Jev: rank the candidates
//!                          │
//!                          ▼
//!                 UI shows verdict + ranked drafts; user copies. Never sends.
//!
//! The judge and draft routes are separate so the browser can show the verdict
//! within about a second instead of waiting for the slower drafting step.
//!
//! # Credentials
//!
//! This process holds no API keys. Every upstream call is paid for by the key
//! the browser sends in `X-Jev-Key`, which lives in that browser's localStorage
//! and nowhere else. The key is held in memory only for the duration of the
//! request it authorises: never written to disk, never logged, never echoed back
//! in a response. Removing the server-side fallback entirely is deliberate — a
//! fallback is a path by which one person's key silently pays for every visitor.

mod questions;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};
use tiny_http::{Header, Method, Request, Response, Server};

const DEFAULT_JEV_BASE: &str = "https://openrouter.ai/api";
const DEFAULT_JEV_MODEL: &str = "jev-latest";
const DEFAULT_DRAFT_BASE: &str = "https://openrouter.ai/api/v1";
const DEFAULT_DRAFT_MODEL: &str = "deepseek/deepseek-chat-v3.1";
// Qwen is trained natively on Chinese and reads chat layouts well, at $0.03/M.
const DEFAULT_VISION_BASE: &str = "https://openrouter.ai/api/v1";
const DEFAULT_VISION_MODEL: &str = "qwen/qwen3.7-flash";
// A pasted screenshot arrives as base64; cap it before spending tokens on it.
const MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;
// A pasted transcript is cheap to send and expensive to process. Long enough for
// a few hundred messages, short enough that a stray whole-file paste is rejected
// before it costs anything.
const MAX_IMPORT_CHARS: usize = 20_000;
/// The browser sends the caller's own OpenRouter key here. Not in the body: the
/// UI renders request payloads in its "raw response" panel, which would put the
/// key on screen.
const KEY_HEADER: &str = "X-Jev-Key";
/// Every upstream call is blocking and can take tens of seconds. A single
/// request loop would let one slow call stall everyone, including the static
/// page, so requests are served by a small pool. They are all IO-bound.
const WORKERS: usize = 8;

/// Endpoints and default model names. Deliberately contains no credentials.
struct Config {
    jev_base: String,
    jev_model: String,
    draft_base: String,
    draft_model: String,
    vision_base: String,
    vision_model: String,
}

#[derive(Deserialize)]
struct JudgeRequest {
    /// Ordered turns: { speaker, text }. Built by the UI, never by a model.
    conversation: Value,
    #[serde(default)]
    background: Option<Value>,
    /// Model override from the settings panel. The base URL is never overridable
    /// from the client: that would turn this into an open relay that forwards
    /// the caller's credentials to an arbitrary host.
    #[serde(default)]
    model: Option<String>,
}

#[derive(Deserialize)]
struct DraftRequest {
    conversation: Value,
    #[serde(default)]
    background: Option<Value>,
    /// The judgement from /api/judge, passed back so drafting respects it.
    verdict: Value,
    /// Drafting model override: from the settings panel, or from the bench.
    #[serde(default)]
    model: Option<String>,
    /// Judge model override, used for the ranking call inside this route.
    #[serde(default)]
    judge_model: Option<String>,
    /// Only the latency bench sets this. The UI never does.
    #[serde(default)]
    reasoning: Option<bool>,
}

#[derive(Deserialize)]
struct VisionRequest {
    /// Full data URL from the browser, e.g. "data:image/png;base64,...".
    image: String,
    /// Optional override, used by the model bench. The UI never sends this.
    #[serde(default)]
    model: Option<String>,
}

#[derive(Deserialize)]
struct ImportRequest {
    /// Raw text copied out of a chat client, newlines and all.
    text: String,
    /// Model override. Defaults to the drafting model: this is a plain text task,
    /// so it has no reason to pay for a vision-capable one.
    #[serde(default)]
    model: Option<String>,
}

fn main() {
    let port = arg_value("--port")
        .or_else(|| std::env::var("JEV_PORT").ok())
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(8778);

    // Default to loopback so running this on a laptop stays private. A container
    // has to opt in explicitly by setting JEV_BIND=0.0.0.0.
    let host = arg_value("--host")
        .or_else(|| std::env::var("JEV_BIND").ok())
        .unwrap_or_else(|| "127.0.0.1".to_string());

    if std::env::args().any(|a| a == "--healthcheck") {
        // Used by the container HEALTHCHECK, which has no shell to curl with.
        std::process::exit(match healthcheck(&host, port) {
            true => 0,
            false => 1,
        });
    }

    let config = Arc::new(load_config());

    let address = format!("{host}:{port}");
    let server = Arc::new(Server::http(&address).unwrap_or_else(|error| {
        eprintln!("could not bind {address}: {error}");
        std::process::exit(1);
    }));

    println!("Jev Copilot   http://{address}  ({WORKERS} workers)");
    println!("judge         {} · {}", config.jev_base, config.jev_model);
    println!("draft         {} · {}", config.draft_base, config.draft_model);
    println!("vision        {} · {}", config.vision_base, config.vision_model);
    println!("keys          bring-your-own, per browser; none stored here");

    let mut workers = Vec::with_capacity(WORKERS);
    for _ in 0..WORKERS {
        let server = Arc::clone(&server);
        let config = Arc::clone(&config);
        workers.push(thread::spawn(move || {
            for request in server.incoming_requests() {
                handle(request, &config);
            }
        }));
    }
    for worker in workers {
        let _ = worker.join();
    }
}

fn arg_value(name: &str) -> Option<String> {
    std::env::args().skip_while(|a| a != name).nth(1)
}

/// Probe our own /healthz. Runs in a throwaway process, so a plain blocking
/// request with a short timeout is all it needs.
fn healthcheck(host: &str, port: u16) -> bool {
    // 0.0.0.0 is a bind address, not a destination.
    let target = if host == "0.0.0.0" || host == "::" { "127.0.0.1" } else { host };
    ureq::get(&format!("http://{target}:{port}/healthz"))
        .timeout(Duration::from_secs(5))
        .call()
        .is_ok()
}

/// Endpoints and default model names, from the environment first and then nearby
/// .env files, so the existing jev-ultrafast / jev-lab setup is reused.
///
/// Note what is *not* read here: no `*_API_KEY`. Startup cannot fail for want of
/// a credential because the server never has one.
fn load_config() -> Config {
    let mut values = std::collections::HashMap::new();
    for name in [
        "TYPESAFE_BASE_URL",
        "TYPESAFE_MODEL",
        "TEXT_MODEL_BASE_URL",
        "TEXT_MODEL",
        "VISION_MODEL_BASE_URL",
        "VISION_MODEL",
    ] {
        if let Ok(value) = std::env::var(name) {
            if !value.is_empty() {
                values.insert(name.to_string(), value);
            }
        }
    }

    for candidate in env_candidates() {
        let Ok(text) = fs::read_to_string(&candidate) else {
            continue;
        };
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with('#') {
                continue;
            }
            if let Some((name, value)) = line.split_once('=') {
                let value = value.trim();
                if !value.is_empty() {
                    values.entry(name.trim().to_string()).or_insert(value.to_string());
                }
            }
        }
    }

    Config {
        jev_base: trim_slash(values.get("TYPESAFE_BASE_URL"), DEFAULT_JEV_BASE),
        jev_model: values
            .get("TYPESAFE_MODEL")
            .cloned()
            .unwrap_or_else(|| DEFAULT_JEV_MODEL.to_string()),
        draft_base: trim_slash(values.get("TEXT_MODEL_BASE_URL"), DEFAULT_DRAFT_BASE),
        draft_model: values
            .get("TEXT_MODEL")
            .cloned()
            .unwrap_or_else(|| DEFAULT_DRAFT_MODEL.to_string()),
        vision_base: trim_slash(values.get("VISION_MODEL_BASE_URL"), DEFAULT_VISION_BASE),
        vision_model: values
            .get("VISION_MODEL")
            .cloned()
            .unwrap_or_else(|| DEFAULT_VISION_MODEL.to_string()),
    }
}

fn trim_slash(value: Option<&String>, fallback: &str) -> String {
    value
        .map(String::as_str)
        .unwrap_or(fallback)
        .trim_end_matches('/')
        .to_string()
}

fn env_candidates() -> Vec<PathBuf> {
    let root = asset_root();
    let parent = root.parent().map(Path::to_path_buf).unwrap_or_else(|| root.clone());
    vec![
        root.join(".env"),
        parent.join("jev-lab").join(".env"),
        parent.join("jev-ultrafast").join(".env"),
    ]
}

/// Assets sit next to the crate in development and next to the exe once shipped.
///
/// In a container neither guess should be load-bearing, so `JEV_ASSET_DIR` wins
/// when set: deployment correctness should not depend on a fallback happening to
/// land in the right place.
fn asset_root() -> PathBuf {
    if let Ok(dir) = std::env::var("JEV_ASSET_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    if manifest.join("index.html").exists() {
        return manifest.to_path_buf();
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| manifest.to_path_buf())
}

/// The caller's own credential, per request. Returned as an owned String that
/// lives no longer than the request it authorises.
fn caller_key(request: &Request) -> Option<String> {
    request
        .headers()
        .iter()
        .find(|header| header.field.equiv(KEY_HEADER))
        .map(|header| header.value.as_str().trim().to_string())
        .filter(|key| !key.is_empty())
}

fn need_key() -> Value {
    json!({
        "error": "需要 API Key",
        "detail": "点右上角「设置」填入你自己的 OpenRouter Key。Key 只存在你的浏览器里。",
        "need_key": true,
    })
}

/// Shared shape of every POST route: require a key, read the body, parse it,
/// dispatch. Factored out so a new route cannot forget the key check.
fn post_route<T: serde::de::DeserializeOwned>(
    mut request: Request,
    handler: impl FnOnce(T, &str) -> (u16, Value),
) {
    let Some(key) = caller_key(&request) else {
        return respond_json(request, 401, &need_key());
    };
    let body = match read_body(&mut request) {
        Ok(body) => body,
        Err(payload) => return respond_json(request, 400, &payload),
    };
    match serde_json::from_str::<T>(&body) {
        Ok(parsed) => {
            let (status, payload) = handler(parsed, &key);
            respond_json(request, status, &payload);
        }
        Err(error) => respond_json(
            request,
            400,
            &json!({"error": "bad request", "detail": error.to_string()}),
        ),
    }
}

fn handle(request: Request, config: &Config) {
    let route = request.url().split('?').next().unwrap_or("/").to_string();

    match (request.method(), route.as_str()) {
        (Method::Get, "/") | (Method::Get, "/index.html") => {
            match fs::read(asset_root().join("index.html")) {
                Ok(bytes) => respond(request, 200, bytes, "text/html; charset=utf-8"),
                Err(error) => respond_json(request, 500, &json!({"error": error.to_string()})),
            }
        }
        (Method::Get, "/healthz") => respond(request, 200, b"ok".to_vec(), "text/plain"),
        (Method::Get, "/api/config") => respond_json(
            request,
            200,
            &json!({
                "judge": { "base": config.jev_base, "model": config.jev_model },
                "draft": { "base": config.draft_base, "model": config.draft_model },
                "vision": { "base": config.vision_base, "model": config.vision_model },
                // Tells the UI to demand a key before enabling anything. There is
                // no server-side fallback, so this is always true; it stays a
                // field rather than a constant so the UI needs no rewrite if that
                // ever changes.
                "byok": true,
            }),
        ),
        (Method::Get, "/api/questions") => {
            respond_json(request, 200, &questions::judgement_questions())
        }
        (Method::Get, "/api/probe") => {
            let Some(key) = caller_key(&request) else {
                return respond_json(request, 401, &need_key());
            };
            let (status, payload) = probe_key(config, &key);
            respond_json(request, status, &payload);
        }
        (Method::Post, "/api/vision") => {
            post_route(request, |parsed: VisionRequest, key| {
                read_screenshot(config, key, parsed)
            })
        }
        (Method::Post, "/api/import") => {
            post_route(request, |parsed: ImportRequest, key| {
                read_transcript(config, key, parsed)
            })
        }
        (Method::Post, "/api/judge") => {
            post_route(request, |parsed: JudgeRequest, key| judge(config, key, parsed))
        }
        (Method::Post, "/api/draft") => {
            post_route(request, |parsed: DraftRequest, key| {
                draft_and_rank(config, key, parsed)
            })
        }
        _ => respond(request, 404, b"not found".to_vec(), "text/plain; charset=utf-8"),
    }
}

/// Check a key without spending tokens, and report the credit headroom that
/// OpenRouter exposes alongside it. Used by the "测试连通" button so a wrong key
/// surfaces in the settings panel instead of as a failed analysis later.
fn probe_key(config: &Config, key: &str) -> (u16, Value) {
    let response = ureq::get(&format!("{}/key", config.draft_base))
        .set("Authorization", &format!("Bearer {key}"))
        .timeout(Duration::from_secs(20))
        .call();

    match response {
        Ok(response) => match response.into_json::<Value>() {
            Ok(value) => {
                let data = value.get("data").unwrap_or(&value);
                (
                    200,
                    json!({
                        "ok": true,
                        "label": data.get("label").and_then(Value::as_str).unwrap_or(""),
                        "usage": data.get("usage").cloned().unwrap_or(Value::Null),
                        "limit": data.get("limit").cloned().unwrap_or(Value::Null),
                        "limit_remaining": data.get("limit_remaining").cloned().unwrap_or(Value::Null),
                        "is_free_tier": data.get("is_free_tier").cloned().unwrap_or(Value::Null),
                    }),
                )
            }
            Err(error) => (
                502,
                json!({"error": "probe body invalid", "detail": error.to_string()}),
            ),
        },
        Err(ureq::Error::Status(401, _)) | Err(ureq::Error::Status(403, _)) => (
            200,
            json!({"ok": false, "error": "Key 无效或已被撤销"}),
        ),
        Err(ureq::Error::Status(code, response)) => (
            200,
            json!({
                "ok": false,
                "error": format!("上游返回 HTTP {code}"),
                "detail": response.into_string().unwrap_or_default(),
            }),
        ),
        Err(error) => (
            200,
            json!({"ok": false, "error": "连接失败", "detail": error.to_string()}),
        ),
    }
}

fn read_body(request: &mut Request) -> Result<String, Value> {
    let mut body = String::new();
    request
        .as_reader()
        .read_to_string(&mut body)
        .map_err(|error| json!({"error": "could not read body", "detail": error.to_string()}))?;
    Ok(body)
}

/// Step 1: seven judgements in a single Jev request.
fn judge(config: &Config, key: &str, parsed: JudgeRequest) -> (u16, Value) {
    let mut state = json!({ "conversation": parsed.conversation });
    if let (Some(object), Some(background)) = (state.as_object_mut(), parsed.background) {
        if !background.is_null() {
            object.insert("background".into(), background);
        }
    }

    let model = parsed
        .model
        .as_deref()
        .filter(|m| !m.is_empty())
        .unwrap_or(&config.jev_model);

    let started = Instant::now();
    match post_jev(config, key, model, &state, &questions::judgement_questions()) {
        Ok(mut value) => {
            if let Some(object) = value.as_object_mut() {
                object.insert("latency_ms".into(), json!(started.elapsed().as_millis()));
            }
            (200, value)
        }
        Err(payload) => (502, payload),
    }
}

/// Step 2 and 3: draft candidates with a chat model, then let Jev rank them.
fn draft_and_rank(config: &Config, key: &str, parsed: DraftRequest) -> (u16, Value) {
    let started = Instant::now();

    let (drafts, draft_usage) = match request_drafts(config, key, &parsed) {
        Ok((drafts, usage)) if !drafts.is_empty() => (drafts, usage),
        Ok(_) => return (502, json!({"error": "drafting model returned no candidates"})),
        Err(payload) => return (502, payload),
    };
    let draft_ms = started.elapsed().as_millis();

    // Ranking is best-effort: candidates are still useful unranked.
    let rank_started = Instant::now();
    let mut ranking = Value::Null;
    let mut rank_error = Value::Null;
    let mut rank_cost = 0.0;

    let mut state = json!({
        "conversation": parsed.conversation,
        "judgement": parsed.verdict,
        "candidates": drafts.iter().enumerate()
            .map(|(index, text)| json!({ "id": format!("draft_{}", index + 1), "text": text }))
            .collect::<Vec<_>>(),
    });
    if let (Some(object), Some(background)) = (state.as_object_mut(), parsed.background.clone()) {
        if !background.is_null() {
            object.insert("background".into(), background);
        }
    }

    let rank_model = parsed
        .judge_model
        .as_deref()
        .filter(|m| !m.is_empty())
        .unwrap_or(&config.jev_model);

    match post_jev(
        config,
        key,
        rank_model,
        &state,
        &questions::ranking_questions(drafts.len()),
    ) {
        Ok(value) => {
            rank_cost = value
                .get("usage")
                .and_then(|usage| usage.get("cost"))
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            ranking = value
                .get("answers")
                .and_then(|answers| answers.get("best_draft"))
                .cloned()
                .unwrap_or(Value::Null);
        }
        Err(payload) => rank_error = payload,
    }

    (
        200,
        json!({
            "drafts": drafts,
            "ranking": ranking,
            "rank_error": rank_error,
            "draft_ms": draft_ms,
            "draft_model": parsed.model.clone().unwrap_or_else(|| config.draft_model.clone()),
            "draft_usage": draft_usage,
            "rank_ms": rank_started.elapsed().as_millis(),
            "rank_cost": rank_cost,
        }),
    )
}

/// Returns the candidates plus the raw usage block, so token composition is
/// visible when tuning latency.
fn request_drafts(
    config: &Config,
    key: &str,
    parsed: &DraftRequest,
) -> Result<(Vec<String>, Value), Value> {
    let user_payload = json!({
        "conversation": parsed.conversation,
        "background": parsed.background,
        "analysis": parsed.verdict,
    });

    let model = parsed
        .model
        .as_deref()
        .filter(|m| !m.is_empty())
        .unwrap_or(&config.draft_model);
    // Three short replies need no chain of thought, and thinking dominated the
    // latency on the vision call for the same reason. Default it off.
    let reasoning = parsed.reasoning.unwrap_or(false);

    let response = ureq::post(&format!("{}/chat/completions", config.draft_base))
        .set("Authorization", &format!("Bearer {key}"))
        .timeout(Duration::from_secs(60))
        .send_json(json!({
            "model": model,
            "max_tokens": 900,
            "response_format": { "type": "json_object" },
            "reasoning": { "enabled": reasoning },
            "messages": [
                { "role": "system", "content": questions::DRAFT_SYSTEM_PROMPT },
                { "role": "user", "content": user_payload.to_string() },
            ],
        }));

    let value: Value = match response {
        Ok(response) => response
            .into_json()
            .map_err(|error| json!({"error": "draft body invalid", "detail": error.to_string()}))?,
        Err(ureq::Error::Status(code, response)) => {
            let detail = response.into_string().unwrap_or_default();
            return Err(json!({"error": format!("draft HTTP {code}"), "detail": detail}));
        }
        Err(error) => {
            return Err(json!({"error": "draft request failed", "detail": error.to_string()}))
        }
    };

    let content = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or_else(|| json!({"error": "draft response had no content", "detail": value.to_string()}))?;

    // The model was asked for JSON, but tolerate a bare array or fenced block.
    let cleaned = content.trim().trim_start_matches("```json").trim_matches('`').trim();
    let parsed_content: Value = serde_json::from_str(cleaned).map_err(|error| {
        json!({"error": "draft content was not JSON", "detail": error.to_string(), "raw": content})
    })?;

    let mut drafts = find_strings(&parsed_content)
        .ok_or_else(|| json!({"error": "draft JSON had no candidate list", "raw": content}))?;
    drafts.truncate(5);
    Ok((drafts, value.get("usage").cloned().unwrap_or(Value::Null)))
}

/// Locate the turn list regardless of how the model wrapped it.
///
/// JSON-mode output from small models is structurally inconsistent. Observed
/// shapes for the same prompt: the requested object, that object inside a
/// single-element array, and the object's keys flattened into an array
/// (`["conversation", [...], "uncertain"]`). Patching each shape is a losing
/// game, so search the tree for the array whose elements carry a `text` field —
/// a signature specific enough not to match anything else in these payloads.
fn find_turns(value: &Value) -> Option<&Vec<Value>> {
    if let Some(array) = value.as_array() {
        if array
            .iter()
            .any(|item| item.get("text").and_then(Value::as_str).is_some())
        {
            return Some(array);
        }
        for item in array {
            if let Some(found) = find_turns(item) {
                return Some(found);
            }
        }
    }
    if let Some(object) = value.as_object() {
        if let Some(found) = object.get("conversation").and_then(find_turns) {
            return Some(found);
        }
        for item in object.values() {
            if let Some(found) = find_turns(item) {
                return Some(found);
            }
        }
    }
    None
}

/// Same idea for drafts: find the first array of non-empty strings.
fn find_strings(value: &Value) -> Option<Vec<String>> {
    if let Some(array) = value.as_array() {
        let strings: Vec<String> = array
            .iter()
            .filter_map(|item| item.as_str().map(str::trim).filter(|s| !s.is_empty()))
            .map(str::to_string)
            .collect();
        // Require at least two so a stray ["conversation"] key list cannot win.
        if strings.len() >= 2 {
            return Some(strings);
        }
        for item in array {
            if let Some(found) = find_strings(item) {
                return Some(found);
            }
        }
    }
    if let Some(object) = value.as_object() {
        if let Some(found) = object.get("drafts").and_then(find_strings) {
            return Some(found);
        }
        for item in object.values() {
            if let Some(found) = find_strings(item) {
                return Some(found);
            }
        }
    }
    None
}

/// Translate the model's `uncertain` line numbers onto the positions the UI
/// renders.
///
/// The model numbers its own output from 1. Blank turns are dropped before the
/// browser sees them, so a raw flag of "line 3" would land on the wrong row.
/// `source_lines[i]` holds the original number of the turn now at position `i`;
/// anything that no longer exists is discarded rather than guessed at.
fn remap_uncertain(parsed_content: &Value, source_lines: &[u64]) -> Vec<Value> {
    let mut seen = Vec::new();
    for line in find_number_array(parsed_content, "uncertain").unwrap_or_default() {
        if let Some(position) = source_lines.iter().position(|source| *source == line) {
            if !seen.contains(&(position + 1)) {
                seen.push(position + 1);
            }
        }
    }
    seen.sort_unstable();
    seen.into_iter().map(|position| json!(position)).collect()
}

/// Pull a named array of line numbers out of the tree wherever the model put it.
///
/// Same structural tolerance as `find_turns`: JSON-mode output wraps keys
/// unpredictably. Numbers sometimes arrive as strings ("2"), so accept both.
fn find_number_array(value: &Value, key: &str) -> Option<Vec<u64>> {
    if let Some(object) = value.as_object() {
        if let Some(array) = object.get(key).and_then(Value::as_array) {
            return Some(
                array
                    .iter()
                    .filter_map(|item| {
                        item.as_u64()
                            .or_else(|| item.as_str().and_then(|s| s.trim().parse().ok()))
                    })
                    .collect(),
            );
        }
        for item in object.values() {
            if let Some(found) = find_number_array(item, key) {
                return Some(found);
            }
        }
    }
    if let Some(array) = value.as_array() {
        for item in array {
            if let Some(found) = find_number_array(item, key) {
                return Some(found);
            }
        }
    }
    None
}

/// Pull a named string out of the tree wherever the model put it.
fn find_string_field(value: &Value, key: &str) -> Option<String> {
    if let Some(object) = value.as_object() {
        if let Some(text) = object.get(key).and_then(Value::as_str) {
            return Some(text.to_string());
        }
        for item in object.values() {
            if let Some(found) = find_string_field(item, key) {
                return Some(found);
            }
        }
    }
    if let Some(array) = value.as_array() {
        for item in array {
            if let Some(found) = find_string_field(item, key) {
                return Some(found);
            }
        }
    }
    None
}

fn post_jev(
    config: &Config,
    key: &str,
    model: &str,
    state: &Value,
    question_set: &Value,
) -> Result<Value, Value> {
    let response = ureq::post(&format!("{}/v1/systemone", config.jev_base))
        .set("Authorization", &format!("Bearer {key}"))
        .timeout(Duration::from_secs(40))
        .send_json(json!({
            "model": model,
            "state": state,
            "questions": question_set,
        }));

    match response {
        Ok(response) => response
            .into_json()
            .map_err(|error| json!({"error": "invalid Jev body", "detail": error.to_string()})),
        // Surface upstream validation text verbatim; it explains schema mistakes.
        Err(ureq::Error::Status(code, response)) => {
            let detail = response.into_string().unwrap_or_default();
            Err(json!({"error": format!("Jev HTTP {code}"), "detail": detail}))
        }
        Err(error) => Err(json!({"error": "Jev request failed", "detail": error.to_string()})),
    }
}

fn respond_json(request: Request, status: u16, payload: &Value) {
    let body = serde_json::to_vec(payload).unwrap_or_else(|_| b"{}".to_vec());
    respond(request, status, body, "application/json; charset=utf-8");
}

fn respond(request: Request, status: u16, body: Vec<u8>, content_type: &str) {
    let mut response = Response::from_data(body).with_status_code(status);
    if let Ok(header) = Header::from_bytes(&b"Content-Type"[..], content_type.as_bytes()) {
        response.add_header(header);
    }
    if let Ok(header) = Header::from_bytes(&b"Cache-Control"[..], &b"no-store"[..]) {
        response.add_header(header);
    }
    let _ = request.respond(response);
}

/// One OpenAI-shaped chat completion. `label` only shapes the error text, so a
/// failure says which step broke.
fn chat_completion(
    base: &str,
    key: &str,
    label: &str,
    timeout: Duration,
    body: Value,
) -> Result<Value, Value> {
    let response = ureq::post(&format!("{base}/chat/completions"))
        .set("Authorization", &format!("Bearer {key}"))
        .timeout(timeout)
        .send_json(body);

    match response {
        Ok(response) => response
            .into_json()
            .map_err(|error| json!({"error": format!("{label} body invalid"), "detail": error.to_string()})),
        Err(ureq::Error::Status(code, response)) => {
            let detail = response.into_string().unwrap_or_default();
            Err(json!({"error": format!("{label} HTTP {code}"), "detail": detail}))
        }
        Err(error) => Err(
            json!({"error": format!("{label} request failed"), "detail": error.to_string()}),
        ),
    }
}

/// Turn a completion into the response both recognition routes return.
///
/// Screenshots and pasted text differ only in how the model is asked; everything
/// after that — unwrapping the JSON, dropping blank turns, translating the
/// `uncertain` flags onto the rows that survive — is identical, and duplicating
/// it would let the two paths drift apart.
fn finish_recognition(
    value: &Value,
    label: &str,
    fallback_model: &str,
    started: Instant,
) -> (u16, Value) {
    let Some(content) = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
    else {
        return (
            502,
            json!({"error": format!("{label} response had no content"), "detail": value.to_string()}),
        );
    };

    let cleaned = content
        .trim()
        .trim_start_matches("```json")
        .trim_matches('`')
        .trim();
    let parsed_content: Value = match serde_json::from_str(cleaned) {
        Ok(parsed) => parsed,
        Err(error) => {
            return (
                502,
                json!({
                    "error": format!("{label} content was not JSON"),
                    "detail": error.to_string(),
                    "raw": content,
                }),
            )
        }
    };

    // Keep each surviving turn's original 1-based position so the model's
    // `uncertain` line numbers can be translated after blank turns are dropped.
    let mut turns: Vec<Value> = Vec::new();
    let mut source_lines: Vec<u64> = Vec::new();
    if let Some(list) = find_turns(&parsed_content) {
        for (index, item) in list.iter().enumerate() {
            let Some(text) = item.get("text").and_then(Value::as_str).map(str::trim) else {
                continue;
            };
            if text.is_empty() {
                continue;
            }
            // Anything other than an explicit "me" is treated as the other
            // party: mislabelling our own line as theirs is the safer error.
            let speaker = match item.get("speaker").and_then(Value::as_str) {
                Some("me") => "me",
                _ => "other",
            };
            turns.push(json!({
                "speaker": speaker,
                "name": item.get("name").and_then(Value::as_str).unwrap_or("").trim(),
                "text": text,
            }));
            source_lines.push(index as u64 + 1);
        }
    }

    if turns.is_empty() {
        return (
            502,
            json!({"error": "没有识别出任何消息", "detail": label, "raw": content}),
        );
    }

    let uncertain = remap_uncertain(&parsed_content, &source_lines);

    (
        200,
        json!({
            "conversation": turns,
            "uncertain": uncertain,
            "note": find_string_field(&parsed_content, "note").unwrap_or_default(),
            "model": value.get("model").cloned().unwrap_or(json!(fallback_model)),
            "usage": value.get("usage").cloned().unwrap_or(Value::Null),
            "latency_ms": started.elapsed().as_millis(),
        }),
    )
}

/// Step 0a: turn a pasted screenshot into structured turns.
///
/// The result is always shown to the user for correction before judging, because
/// speaker attribution from bubble geometry is the part most likely to be wrong.
fn read_screenshot(config: &Config, key: &str, parsed: VisionRequest) -> (u16, Value) {
    let image = parsed.image.trim();
    if !image.starts_with("data:image/") {
        return (
            400,
            json!({"error": "expected a data:image/... URL from the browser"}),
        );
    }
    // base64 inflates by 4/3; this is a cheap guard, not an exact byte count.
    if image.len() / 4 * 3 > MAX_IMAGE_BYTES {
        return (
            413,
            json!({"error": "截图太大", "detail": "控制在 8 MB 以内"}),
        );
    }

    let model = parsed
        .model
        .as_deref()
        .filter(|m| !m.is_empty())
        .unwrap_or(&config.vision_model);

    let started = Instant::now();
    let value = match chat_completion(
        &config.vision_base,
        key,
        "vision",
        Duration::from_secs(90),
        json!({
            "model": model,
            "max_tokens": 2400,
            "response_format": { "type": "json_object" },
            // Transcribing a screenshot is perception, not deliberation. The first
            // run burned 203 reasoning tokens on it, so turn thinking off: it costs
            // latency and money without improving what comes back.
            "reasoning": { "enabled": false },
            "messages": [
                { "role": "system", "content": questions::VISION_SYSTEM_PROMPT },
                { "role": "user", "content": [
                    { "type": "text", "text": "请读出这张截图里的对话。" },
                    { "type": "image_url", "image_url": { "url": image } }
                ]}
            ],
        }),
    ) {
        Ok(value) => value,
        Err(payload) => return (502, payload),
    };

    finish_recognition(&value, "vision", model, started)
}

/// Step 0b: split a pasted transcript into structured turns.
///
/// Same output contract as `read_screenshot` so the browser has one rendering
/// path for both. Runs on the text model: splitting a transcript needs no vision
/// capability and no chain of thought.
fn read_transcript(config: &Config, key: &str, parsed: ImportRequest) -> (u16, Value) {
    let text = parsed.text.trim();
    if text.is_empty() {
        return (400, json!({"error": "没有内容", "detail": "先粘贴一段聊天记录"}));
    }
    if text.chars().count() > MAX_IMPORT_CHARS {
        return (
            413,
            json!({
                "error": "内容太长",
                "detail": format!("控制在 {MAX_IMPORT_CHARS} 字以内，只贴相关的那一段就够了"),
            }),
        );
    }

    let model = parsed
        .model
        .as_deref()
        .filter(|m| !m.is_empty())
        .unwrap_or(&config.draft_model);

    let started = Instant::now();
    let value = match chat_completion(
        &config.draft_base,
        key,
        "import",
        Duration::from_secs(60),
        json!({
            "model": model,
            "max_tokens": 4000,
            "response_format": { "type": "json_object" },
            "reasoning": { "enabled": false },
            "messages": [
                { "role": "system", "content": questions::IMPORT_SYSTEM_PROMPT },
                { "role": "user", "content": text },
            ],
        }),
    ) {
        Ok(value) => value,
        Err(payload) => return (502, payload),
    };

    finish_recognition(&value, "import", model, started)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Requirement: the flags the vision model raises must reach the UI, whatever
    /// shape JSON mode wrapped them in.
    #[test]
    fn uncertain_is_found_through_the_usual_wrappers() {
        let plain = json!({"conversation": [], "uncertain": [2, 4]});
        assert_eq!(find_number_array(&plain, "uncertain"), Some(vec![2, 4]));

        let wrapped_in_array = json!([{"conversation": [], "uncertain": [1]}]);
        assert_eq!(find_number_array(&wrapped_in_array, "uncertain"), Some(vec![1]));

        let nested = json!({"result": {"data": {"uncertain": [3]}}});
        assert_eq!(find_number_array(&nested, "uncertain"), Some(vec![3]));

        // Some models quote the numbers.
        let stringly = json!({"uncertain": ["2", "3"]});
        assert_eq!(find_number_array(&stringly, "uncertain"), Some(vec![2, 3]));

        // No flags at all is the common case and must not be an error.
        assert_eq!(find_number_array(&json!({"conversation": []}), "uncertain"), None);
    }

    /// Requirement: a flag must highlight the row the user actually sees. Blank
    /// turns are dropped, so the numbering has to shift with them.
    #[test]
    fn uncertain_follows_the_rows_that_survived() {
        // Model emitted 5 turns; turns 2 and 4 were blank and got dropped, so the
        // rendered rows came from original lines 1, 3, 5.
        let source_lines = [1, 3, 5];
        let content = json!({"uncertain": [3, 5]});
        assert_eq!(
            remap_uncertain(&content, &source_lines),
            vec![json!(2), json!(3)]
        );
    }

    #[test]
    fn uncertain_drops_flags_for_rows_that_no_longer_exist() {
        let source_lines = [1, 3];
        // Line 2 was blank and dropped; flagging it must not mark an unrelated row.
        let content = json!({"uncertain": [2]});
        assert!(remap_uncertain(&content, &source_lines).is_empty());
    }

    #[test]
    fn uncertain_is_sorted_and_deduplicated() {
        let source_lines = [1, 2, 3];
        let content = json!({"uncertain": [3, 1, 3]});
        assert_eq!(
            remap_uncertain(&content, &source_lines),
            vec![json!(1), json!(3)]
        );
    }

    /// Requirement: an out-of-range flag must never be handed to the browser,
    /// because the UI turns it into "check line N" for a line that is not there.
    #[test]
    fn uncertain_ignores_out_of_range_flags() {
        let source_lines = [1, 2];
        let content = json!({"uncertain": [0, 9]});
        assert!(remap_uncertain(&content, &source_lines).is_empty());
    }
}
