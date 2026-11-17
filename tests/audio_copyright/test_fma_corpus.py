"""
Preparación del corpus de audio (apps/audio-copyright/dataset). Usa zips falsos con el
mismo formato que los de FMA, así que no necesita red.
"""

import csv
import io
import sys
import zipfile
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[2] / "apps" / "audio-copyright" / "dataset"
sys.path.insert(0, str(DATA_DIR))

import prepare_corpus  # noqa: E402
from fma_corpus import (  # noqa: E402
    Track,
    candidates_by_genre,
    file_name,
    license_family,
    member_name,
    read_excluded,
    read_manifest,
    read_small_tracks,
)
from prepare_corpus import FMA_SMALL_GENRES, prepare  # noqa: E402

BY_NC_SA = "http://creativecommons.org/licenses/by-nc-sa/3.0/"
BY_NC_ND = "http://creativecommons.org/licenses/by-nc-nd/4.0/"


# Licencias

@pytest.mark.parametrize("url, expected", [
    ("http://creativecommons.org/licenses/by/4.0/", "CC BY"),
    ("http://creativecommons.org/licenses/by-sa/3.0/us/", "CC BY-SA"),
    ("https://creativecommons.org/licenses/by-nc/3.0/de/", "CC BY-NC"),
    (BY_NC_SA, "CC BY-NC-SA"),
    ("http://creativecommons.org/publicdomain/zero/1.0/", "CC0"),
    ("http://creativecommons.org/publicdomain/mark/1.0/", "Public Domain"),
    ("http://creativecommons.org/licenses/publicdomain/", "Public Domain"),
])
def test_licenses_that_allow_derivatives_are_accepted(url, expected):
    assert license_family(url) == expected


@pytest.mark.parametrize("url", [
    BY_NC_ND,
    "http://creativecommons.org/licenses/by-nd/3.0/",
    "http://creativecommons.org/licenses/sampling+/1.0/",
    "http://creativecommons.org/licenses/nc-sampling+/1.0/",
    "http://artlibre.org/licence/lal/en",
    "",
])
def test_no_derivatives_sampling_and_unknown_licenses_are_rejected(url):
    assert license_family(url) is None


# Nombres y exclusiones

def test_member_and_file_names_follow_fma_but_the_corpus_is_flat():
    assert member_name(2) == "fma_small/000/000002.mp3"
    assert member_name(155066) == "fma_small/155/155066.mp3"
    assert file_name(155066) == "155066.mp3"


def test_excluded_tracks_ignore_comments_and_blank_lines(tmp_path):
    path = tmp_path / "excluded.txt"
    path.write_text("# encabezado\n\n12  # retirada por el artista\n 34\n", encoding="utf-8")

    assert read_excluded(path) == {12, 34}
    assert read_excluded(tmp_path / "no_existe.txt") == set()


# Selección

def track(song_id, genre="Rock", url=BY_NC_SA):
    return Track(song_id, f"Canción {song_id}", song_id, f"Artista {song_id}", "Álbum", genre,
                 "licencia", url, license_family(url), f"http://fma/{song_id}", f"http://fma/a/{song_id}")


def test_candidates_skip_rejected_licenses_and_excluded_tracks_in_a_fixed_order():
    tracks = [track(i) for i in range(1, 21)] + [track(99, url=BY_NC_ND)]

    first = candidates_by_genre(tracks, excluded={5})
    second = candidates_by_genre(list(reversed(tracks)), excluded={5})

    ids = [t.song_id for t in first["Rock"]]
    assert ids == [t.song_id for t in second["Rock"]]
    assert 99 not in ids and 5 not in ids
    assert sorted(ids) == [i for i in range(1, 21) if i != 5]
    assert ids != sorted(ids)


# Metadatos y descarga con zips falsos

def write_metadata_zip(path, tracks):
    """tracks.csv con dos filas de encabezado, como el de FMA, y raw_tracks.csv."""
    groups = ["", "album", "artist", "artist", "set", "track", "track"]
    columns = ["", "title", "id", "name", "subset", "genre_top", "title"]
    tracks_csv = io.StringIO()
    writer = csv.writer(tracks_csv)
    writer.writerows([groups, columns, ["track_id"] + [""] * 6])
    for t, subset in tracks:
        writer.writerow([t.song_id, t.album_title, t.artist_id, t.artist_name, subset, t.genre, t.title])
    raw_csv = io.StringIO()
    writer = csv.writer(raw_csv)
    writer.writerow(["track_id", "license_title", "license_url", "track_url", "artist_url"])
    for t, _ in tracks:
        writer.writerow([t.song_id, t.license_title, t.license_url, t.track_url, t.artist_url])
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fma_metadata/tracks.csv", tracks_csv.getvalue())
        archive.writestr("fma_metadata/raw_tracks.csv", raw_csv.getvalue())


def write_audio_zip(path, song_ids):
    with zipfile.ZipFile(path, "w") as archive:
        for song_id in song_ids:
            archive.writestr(member_name(song_id), f"audio {song_id}".encode() * 10)


@pytest.fixture
def fake_fma(tmp_path):
    """3 pistas aceptables por género, una ND y una de otro subconjunto."""
    tracks = [(track(genre_index * 10 + i, genre), "small")
              for genre_index, genre in enumerate(FMA_SMALL_GENRES) for i in range(1, 4)]
    tracks += [(track(500, "Rock", BY_NC_ND), "small"), (track(600, "Pop"), "medium")]
    metadata, audio = tmp_path / "fma_metadata.zip", tmp_path / "fma_small.zip"
    write_metadata_zip(metadata, tracks)
    write_audio_zip(audio, [t.song_id for t, _ in tracks])
    return metadata, audio


def test_small_tracks_join_titles_with_licenses_and_urls(fake_fma):
    metadata, _ = fake_fma
    with zipfile.ZipFile(metadata) as archive:
        tracks = read_small_tracks(archive)

    assert len(tracks) == 3 * len(FMA_SMALL_GENRES) + 1     # sin la de "medium"
    first = tracks[0]
    assert (first.title, first.artist_name, first.license_family) == (
        f"Canción {first.song_id}", f"Artista {first.song_id}", "CC BY-NC-SA")
    assert next(t for t in tracks if t.song_id == 500).license_family is None


def run(fake_fma, corpus, per_genre=2, durations=None, excluded=None):
    metadata, audio = fake_fma
    durations = durations or {}
    return prepare(
        corpus, per_genre=per_genre, workers=2, metadata_source=str(metadata), audio_source=str(audio),
        excluded_path=excluded or corpus.parent / "sin_exclusiones.txt",
        duration_of=lambda path: durations.get(int(path.stem), 30.0),
    )


def test_prepare_downloads_the_quota_of_every_genre_with_its_metadata(fake_fma, tmp_path):
    corpus = tmp_path / "corpus"

    summary = run(fake_fma, corpus)

    songs = read_manifest(corpus / "songs.csv")
    assert len(songs) == 2 * len(FMA_SMALL_GENRES)
    assert summary["by_genre"] == {genre: 2 for genre in FMA_SMALL_GENRES}
    assert 500 not in songs and 600 not in songs
    assert sorted(p.name for p in corpus.glob("*.mp3")) == sorted(file_name(i) for i in songs)
    assert (corpus / "corpus_summary.json").exists()
    song = next(iter(songs.values()))
    assert song.source == "fma_small" and song.license_url == BY_NC_SA and len(song.sha256) == 64


def test_a_second_run_downloads_nothing(fake_fma, tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    run(fake_fma, corpus)
    monkeypatch.setattr(prepare_corpus, "fetch_song", lambda *args, **kwargs: pytest.fail("no debería descargar"))

    summary = run(fake_fma, corpus)

    assert summary["downloaded_bytes"] == 0
    assert summary["songs"] == 2 * len(FMA_SMALL_GENRES)


def test_a_damaged_file_is_downloaded_again(fake_fma, tmp_path):
    corpus = tmp_path / "corpus"
    run(fake_fma, corpus)
    damaged = sorted(corpus.glob("*.mp3"))[0]
    damaged.write_bytes(b"roto")

    run(fake_fma, corpus)

    song = read_manifest(corpus / "songs.csv")[int(damaged.stem)]
    assert damaged.stat().st_size == song.size_bytes


def test_a_short_clip_is_replaced_by_the_next_candidate(fake_fma, tmp_path):
    corpus = tmp_path / "corpus"
    first_run = read_manifest_after(run, fake_fma, corpus)
    short = next(iter(first_run))

    summary = run(fake_fma, tmp_path / "otro", durations={short: 3.0})

    assert short in summary["rejected"]
    assert short not in read_manifest(tmp_path / "otro" / "songs.csv")
    assert summary["songs"] == 2 * len(FMA_SMALL_GENRES)


def test_an_excluded_track_is_removed_from_the_corpus(fake_fma, tmp_path):
    corpus = tmp_path / "corpus"
    removed = next(iter(read_manifest_after(run, fake_fma, corpus)))
    excluded = tmp_path / "excluded.txt"
    excluded.write_text(f"{removed}\n", encoding="utf-8")

    run(fake_fma, corpus, excluded=excluded)

    assert removed not in read_manifest(corpus / "songs.csv")
    assert not (corpus / file_name(removed)).exists()


def test_not_enough_licensed_tracks_is_an_error(fake_fma, tmp_path):
    with pytest.raises(RuntimeError, match="No hay 4 pistas"):
        run(fake_fma, tmp_path / "corpus", per_genre=4)


def read_manifest_after(runner, fake_fma, corpus):
    runner(fake_fma, corpus)
    return read_manifest(corpus / "songs.csv")
