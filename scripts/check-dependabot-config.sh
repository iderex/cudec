#!/bin/sh
# Refuses a .github/dependabot.yml the platform would reject.
#
# The failure this prevents is silent: GitHub parses the file on its own
# schedule and reports a rejection nowhere this repository can read, so a
# misspelled key sits in the tree looking configured while the setting it
# names is not in force. Prettier already reds an unparseable file - it is
# the YAML parser in the format job - and passes a well-formed file whose
# keys the platform does not know, which is the whole gap this closes.
#
# Route: validate against the published JSON schema. The other route the
# issue named, reading the platform's own parse status, has no public API -
# GitHub's REST surface for Dependabot is alerts and secrets, and the
# validation result appears only in the web UI. So the schema is the only
# machine-readable statement of the config grammar available, and its known
# limit is stated rather than glossed: Dependabot applies checks beyond the
# schema, so a file this accepts can still be refused upstream. It is a floor.
#
# The schema is fetched under a SHA-256 pin rather than copied into the tree,
# which is the pattern the masterplan's oracle-pinning policy already sets: a
# copy restates bytes in a form that no longer carries its own verification.
# When upstream edits the schema the pin mismatches and this reds - that is
# the invariant working, and the repair is to read the new schema and move the
# pin deliberately.
#
# Fail-closed at every step, because a config check that checked nothing reads
# exactly like a clean one: an absent config file, a failed fetch, a hash
# mismatch, an unparseable file, and a validator that cannot refuse a known
# bad document are all hard errors here and never skips.
set -eu

cfg="${1:-.github/dependabot.yml}"

# Pinned 2026-08-31 against https://www.schemastore.org/dependabot-2.0.json
# (json.schemastore.org 301-redirects there). Recompute with:
#   curl -fsSL "$schema_url" | sha256sum
#
# Moved from 46255f69 on 2026-08-31 after reading what changed upstream (#422).
# One commit touched the file in the window, SchemaStore 07a2d101 "make
# multi-ecosystem patterns optional", and its whole diff is the removal of one
# conditional: a config naming multi-ecosystem-group no longer has to name
# patterns. That is a loosening, so it cannot invalidate a document that
# validated before it, and this repository uses neither key - so the rule that
# went away could never have applied here.
schema_url="https://www.schemastore.org/dependabot-2.0.json"
schema_sha256="c6c2432adc55b40f828b0f987f8024bb9b8926a978301e9cdf4fe3410ee7cf31"

# Exact versions, so the gate's own dependencies cannot move underneath it.
yaml_pkg="js-yaml@4.1.0"
ajv_pkg="ajv-cli@5.0.0"

die() {
    echo "check-dependabot-config: $1" >&2
    exit 1
}

command -v curl >/dev/null 2>&1 || die "curl is not on PATH"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is not on PATH"
command -v npx >/dev/null 2>&1 || die "npx is not on PATH"

[ -f "$cfg" ] ||
    die "$cfg does not exist. Deleting or renaming the config is the silent
failure this check exists for, so an absent file is refused rather than
skipped."

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

curl -fsSL "$schema_url" -o "$work/schema.json" ||
    die "could not fetch $schema_url"
echo "$schema_sha256  $work/schema.json" | sha256sum -c - >/dev/null ||
    die "the schema at $schema_url does not match the recorded SHA-256.
Read what changed upstream, then move the pin in this file deliberately."

# ajv reads JSON, the config is YAML, and js-yaml is the conversion. A parse
# failure here is the malformed-file case and stops the run.
to_json() {
    npx --yes --package="$yaml_pkg" -- js-yaml "$1" >"$2" ||
        die "$1 is not parseable as YAML"
}

validate() {
    npx --yes --package="$ajv_pkg" -- ajv validate \
        --spec=draft7 --strict=false -s "$work/schema.json" -d "$1"
}

# The validator proves itself on every run before it is trusted with the real
# file. A wrong flag, a wrong path, or a schema that lost its
# additionalProperties clause would otherwise pass everything silently, which
# is the vacuous run in another shape. The planted defect is the one the
# platform actually rejects: an underscore where the key takes a hyphen.
cat >"$work/self-test.yml" <<'EOF'
version: 2
updates:
  - package-ecosystem: github-actions
    directory: "/"
    schedule:
      interval: weekly
    cooldown:
      default_days: 7
EOF
to_json "$work/self-test.yml" "$work/self-test.json"
if validate "$work/self-test.json" >"$work/self-test.log" 2>&1; then
    cat "$work/self-test.log" >&2
    die "the self-test document was accepted. It carries default_days where
the schema takes default-days, so a validator that accepts it is not
refusing anything and its verdict on the real config means nothing."
fi
echo "PASS: self-test - the validator refuses a known-bad config"

to_json "$cfg" "$work/config.json"
validate "$work/config.json" ||
    die "$cfg does not validate against the published Dependabot schema.
The report above names the key. A config the schema refuses is one the
platform is likely to reject, and a rejected config is not in force."

echo "PASS: $cfg validates against $schema_url"
