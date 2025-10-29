-- Remember Twelve Database Schema
-- SQLite database for photo curation and year-in-review generation
-- Location: ~/.remember_twelve/remember_twelve.db

-- Schema versioning for migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Photos table: stores all imported photos with scoring
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    source_path TEXT,
    captured_at TIMESTAMP NOT NULL,
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    month INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
    year INTEGER NOT NULL CHECK (year >= 1900),

    quality_score REAL CHECK (quality_score >= 0 AND quality_score <= 100),
    emotional_score REAL CHECK (emotional_score >= 0 AND emotional_score <= 100),
    combined_score REAL CHECK (combined_score >= 0 AND combined_score <= 100),

    metadata_json TEXT,

    UNIQUE(filename, year)
);

-- Curations table: tracks curation runs
CREATE TABLE IF NOT EXISTS curations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1,
    stats_json TEXT
);

-- Curation photos: the twelve selected photos per curation
CREATE TABLE IF NOT EXISTS curation_photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    curation_id INTEGER NOT NULL,
    photo_id INTEGER NOT NULL,
    month_slot INTEGER NOT NULL CHECK (month_slot BETWEEN 1 AND 12),

    FOREIGN KEY (curation_id) REFERENCES curations(id) ON DELETE CASCADE,
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
    UNIQUE(curation_id, month_slot)
);

-- Swaps table: audit log of manual swaps
CREATE TABLE IF NOT EXISTS swaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    curation_id INTEGER NOT NULL,
    month_slot INTEGER NOT NULL,
    old_photo_id INTEGER NOT NULL,
    new_photo_id INTEGER NOT NULL,
    swapped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reason TEXT,

    FOREIGN KEY (curation_id) REFERENCES curations(id) ON DELETE CASCADE,
    FOREIGN KEY (old_photo_id) REFERENCES photos(id),
    FOREIGN KEY (new_photo_id) REFERENCES photos(id)
);

-- Indexes for common queries

-- Find photos by year/month for curation
CREATE INDEX IF NOT EXISTS idx_photos_year_month
    ON photos(year, month);

-- Find photos by combined score for ranking
CREATE INDEX IF NOT EXISTS idx_photos_score
    ON photos(year, combined_score DESC);

-- Find active curation for a year
CREATE INDEX IF NOT EXISTS idx_curations_year_active
    ON curations(year, is_active);

-- Find curation photos by curation
CREATE INDEX IF NOT EXISTS idx_curation_photos_curation
    ON curation_photos(curation_id);

-- Find swaps by curation
CREATE INDEX IF NOT EXISTS idx_swaps_curation
    ON swaps(curation_id);

-- Sample Queries

-- Get top 12 photos for a year by combined score:
-- SELECT * FROM photos
-- WHERE year = 2023
-- ORDER BY combined_score DESC
-- LIMIT 12;

-- Get photos for specific month:
-- SELECT * FROM photos
-- WHERE year = 2023 AND month = 6
-- ORDER BY combined_score DESC;

-- Get month distribution for year:
-- SELECT month, COUNT(*) as photo_count
-- FROM photos
-- WHERE year = 2023
-- GROUP BY month
-- ORDER BY month;

-- Get the twelve photos for active curation:
-- SELECT p.*, cp.month_slot
-- FROM curation_photos cp
-- JOIN photos p ON cp.photo_id = p.id
-- JOIN curations c ON cp.curation_id = c.id
-- WHERE c.year = 2023 AND c.is_active = 1
-- ORDER BY cp.month_slot;

-- Get swap history for a curation:
-- SELECT s.*,
--        old.filename as old_photo,
--        new.filename as new_photo
-- FROM swaps s
-- JOIN photos old ON s.old_photo_id = old.id
-- JOIN photos new ON s.new_photo_id = new.id
-- WHERE s.curation_id = ?
-- ORDER BY s.swapped_at DESC;

-- Get curation statistics:
-- SELECT
--     c.year,
--     c.strategy,
--     c.created_at,
--     COUNT(cp.id) as photo_count,
--     AVG(p.combined_score) as avg_score,
--     COUNT(s.id) as swap_count
-- FROM curations c
-- LEFT JOIN curation_photos cp ON c.id = cp.curation_id
-- LEFT JOIN photos p ON cp.photo_id = p.id
-- LEFT JOIN swaps s ON c.id = s.curation_id
-- WHERE c.year = 2023 AND c.is_active = 1
-- GROUP BY c.id;
