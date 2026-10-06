-- =====================================================================
-- CNC MACHINE PREDICTIVE MAINTENANCE SYSTEM
-- Clean & Simple Database Schema for MySQL / XAMPP
-- Database: cnc_maintenance_db
-- =====================================================================

CREATE DATABASE IF NOT EXISTS cnc_maintenance_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE cnc_maintenance_db;

-- ---------------------------------------------------------------------
-- 1. Table: users (Authentication and System Operators)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(150) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'Maintenance Engineer',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------
-- 2. Table: machines (CNC Machine Asset Inventory - Tracked by Machine ID)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS machines (
    id INT AUTO_INCREMENT PRIMARY KEY,
    machine_code VARCHAR(50) NOT NULL UNIQUE,
    machine_name VARCHAR(150) NULL DEFAULT NULL,
    location VARCHAR(150) NULL DEFAULT '',
    description TEXT,
    status VARCHAR(50) DEFAULT 'Healthy',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------
-- 3. Table: predictions (Historical Predictive Maintenance Inferences)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    machine_id INT NOT NULL,
    model_used VARCHAR(80) NOT NULL DEFAULT 'SMOTE-XGBoost',
    vibration_rms DECIMAL(8, 4) NOT NULL,
    temperature_motor DECIMAL(8, 4) NOT NULL,
    current_phase_avg DECIMAL(8, 4) NOT NULL,
    pressure_level DECIMAL(8, 4) NOT NULL,
    rpm DECIMAL(8, 4) NOT NULL,
    prediction TINYINT NOT NULL COMMENT '0 = No Failure within 24h, 1 = Failure within 24h',
    failure_probability DECIMAL(6, 2) NOT NULL COMMENT 'Calculated probability percentage',
    risk_level VARCHAR(30) NOT NULL COMMENT 'Low Risk, Medium Risk, High Risk',
    recommendation TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_machine_id (machine_id),
    INDEX idx_prediction (prediction),
    INDEX idx_risk_level (risk_level),
    INDEX idx_created_at (created_at),
    FOREIGN KEY (machine_id) REFERENCES machines(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------
-- 4. Initial Seed Data (Only Admin User - Tables are kept clean and empty)
-- ---------------------------------------------------------------------

-- Insert Default Admin: admin / admin123
INSERT IGNORE INTO users (id, username, email, password_hash, role) VALUES
(1, 'admin', 'admin@cnc.com', 'scrypt:32768:8:1$d4cuoyucUu7QoRJi$931c086cc663efbfab0866029e2332db5938573b19186886af1029209db132e5d3e35cd0828165da3e6fac64fc5d2b83a204610c4823ce6dd647e76bc8f1704d', 'Maintenance Engineer');
