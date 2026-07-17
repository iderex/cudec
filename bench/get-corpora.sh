#!/bin/sh
# Fetches the benchmark corpora into bench/corpora/ (git-ignored; the
# supply-chain rule applied to data): SHA-256-pinned, fail-closed on any
# mismatch, never committed. Needs curl and unzip.
#
# Two mirrors per corpus. The Silesia mirrors ship differently-packaged
# zips, so each mirror carries its own zip hash; the twelve contained
# corpus files were verified byte-identical across both packagings at pin
# time (2026-07-17). enwik8 is byte-identical on both mirrors.
set -eu

dir="$(dirname "$0")/corpora"
mkdir -p "$dir"

fetch() {
    name="$1"
    sum="$2"
    url="$3"
    out="$dir/$name"
    if [ -f "$out" ] && echo "$sum  $out" | sha256sum -c --status; then
        echo "$name: present and verified"
        return 0
    fi
    echo "$name: fetching $url"
    curl -fSL --retry 3 -o "$out.tmp" "$url" || return 1
    if ! echo "$sum  $out.tmp" | sha256sum -c --status; then
        rm -f "$out.tmp"
        echo "$name: SHA-256 mismatch from $url - refusing the file" >&2
        return 1
    fi
    mv "$out.tmp" "$out"
}

fetch enwik8.zip \
    547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc \
    "http://mattmahoney.net/dc/enwik8.zip" ||
    fetch enwik8.zip \
        547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc \
        "https://data.deepai.org/enwik8.zip"

fetch silesia.zip \
    0626e25f45c0ffb5dc801f13b7c82a3b75743ba07e3a71835a41e3d9f63c77af \
    "http://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip" ||
    fetch silesia.zip \
        7d1dd71bfecda66a0ca30d863ed031809f67ecf12717a60fe72c1cc39e28434e \
        "http://mattmahoney.net/dc/silesia.zip"

unzip -oq "$dir/enwik8.zip" -d "$dir"
unzip -oq "$dir/silesia.zip" -d "$dir/silesia"

# The zip pins guarantee the archives; this manifest pins the FILES the
# harness actually reads - packaging-independent (the mirrors' archives
# differ), and it catches a half-extracted tree left by an interrupt.
cat >"$dir/manifest.sha256" <<'EOF'
2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8  enwik8
b24c37886142e11d0ee687db6ab06f936207aa7f2ea1fd1d9a36763c7a507e6a  silesia/dickens
657fc3764b0c75ac9de9623125705831ebbfbe08fed248df73bc2dc66e2a963b  silesia/mozilla
68637ed52e3e4860174ed2dc0840ac77d5f1a60abbcb13770d5754e3774d53e6  silesia/mr
fc63a31770947b8c2062d3b19ca94c00485a232bb91b502021948fee983e1635  silesia/nci
e7ee013880d34dd5208283d0d3d91b07f442e067454276095ded14f322a656eb  silesia/ooffice
60f027179302ca3ad87c58ac90b6be72ec23588aaa7a3b7fe8ecc0f11def3fa3  silesia/osdb
0eac0114a3dfe6e2ee1f345a0f79d653cb26c3bc9f0ed79238af4933422b7578  silesia/reymont
93ba07bc44d8267789c1d911992f40b089ffa2140b4a160fac11ccae9a40e7b2  silesia/samba
c2d0ea2cc59d4c21b7fe43a71499342a00cbe530a1d5548770e91ecd6214adcc  silesia/sao
6a68f69b26daf09f9dd84f7470368553194a0b294fcfa80f1604efb11143a383  silesia/webster
7de9fce1405dc44ae5e6813ed21cd5751e761bd4265655a005d39b9685d1c9ad  silesia/x-ray
0e82e54e695c1938e4193448022543845b33020c8be6bf3bf3ead2224903e08c  silesia/xml
EOF
(cd "$dir" && sha256sum -c --quiet manifest.sha256)
echo "corpora ready under $dir"
