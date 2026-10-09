-- Optional MySQL backend. The CSV in data/ is the default source.
-- Load:  mysql -u <user> -p < init_experiments.sql   (run from the repo root)
-- Use:   YIELD_DATA_SOURCE=mysql YIELD_DB_USER=<user> YIELD_DB_PASSWORD=<pw> python train.py

CREATE DATABASE IF NOT EXISTS experiments;
USE experiments;

CREATE TABLE IF NOT EXISTS data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    temperature FLOAT NOT NULL,            -- deg C
    pressure FLOAT NOT NULL,               -- atm
    catalyst_concentration FLOAT NOT NULL, -- mol/L
    yield_pct FLOAT NOT NULL CHECK (yield_pct BETWEEN 0 AND 100)
);

LOAD DATA LOCAL INFILE 'data/chemical_reaction_yield_dataset.csv'
INTO TABLE data
FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
IGNORE 1 LINES
(temperature, pressure, catalyst_concentration, yield_pct);
