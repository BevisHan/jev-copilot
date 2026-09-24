# syntax=docker/dockerfile:1

# ---- build ------------------------------------------------------------------
# Pinned to bookworm on purpose. The runtime image below is Debian 12 based, and
# a newer Debian builder would link against a glibc that runtime does not have.
FROM rust:1.98-slim-bookworm AS build
WORKDIR /src

# Compile the dependency graph on its own layer, so editing src/ does not
# re-download and rebuild the whole tree every time. The dummy main is enough:
# it pulls in Cargo.toml's dependencies without needing the real sources.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src \
 && echo 'fn main() {}' > src/main.rs \
 && cargo build --release \
 && rm -rf src

COPY src ./src
# COPY already refreshes mtimes, but cargo decides by timestamp, so be explicit.
RUN touch src/main.rs && cargo build --release

# ---- runtime ----------------------------------------------------------------
# distroless/cc rather than scratch: `ring` (pulled in by rustls) needs libc.
#
# Note what is absent: no ca-certificates. ureq here uses rustls with
# webpki-roots, so the trust store is compiled into the binary.
FROM gcr.io/distroless/cc-debian12
WORKDIR /app

COPY --from=build /src/target/release/jev-copilot /app/jev-copilot
COPY index.html /app/index.html

# The binary defaults to 127.0.0.1 so running it on a laptop stays private;
# a container has to opt in to a public interface explicitly.
#
# JEV_ASSET_DIR is not optional here. asset_root() would otherwise fall back to
# guessing, and deployment correctness should not rest on a fallback landing in
# the right place.
ENV JEV_BIND=0.0.0.0 \
    JEV_PORT=8778 \
    JEV_ASSET_DIR=/app

EXPOSE 8778
USER nonroot

# distroless ships no shell and no curl, so the check runs the binary's own
# probe mode against /healthz.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD ["/app/jev-copilot", "--healthcheck"]

ENTRYPOINT ["/app/jev-copilot"]
