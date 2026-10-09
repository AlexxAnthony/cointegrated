-- Database layout used by daily_update.py (PostgreSQL with the TimescaleDB extension).
-- The signals and paper_portfolio tables are created automatically by the daily job.

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS prices (
    symbol    TEXT NOT NULL,
    date      DATE NOT NULL,
    open      DOUBLE PRECISION,
    high      DOUBLE PRECISION,
    low       DOUBLE PRECISION,
    close     DOUBLE PRECISION,
    adj_close DOUBLE PRECISION,
    volume    BIGINT,
    PRIMARY KEY (symbol, date)
);

SELECT create_hypertable('prices', 'date', if_not_exists => TRUE);
