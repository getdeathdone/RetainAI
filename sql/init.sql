-- RetainAI: PostgreSQL initialization script.
-- Creates a realistic synthetic customer activity dataset for churn/LTV modeling.

DROP SCHEMA IF EXISTS retainai CASCADE;
CREATE SCHEMA retainai;
SET search_path TO retainai;

SELECT setseed(0.42);

CREATE TABLE users (
    user_id BIGINT PRIMARY KEY,
    signup_ts TIMESTAMPTZ NOT NULL,
    country TEXT NOT NULL,
    acquisition_channel TEXT NOT NULL,
    plan_type TEXT NOT NULL,
    marketing_opt_in BOOLEAN NOT NULL,
    age INTEGER CHECK (age BETWEEN 16 AND 90),
    gender TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE transactions (
    transaction_id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    transaction_ts TIMESTAMPTZ NOT NULL,
    amount NUMERIC(10, 2) NOT NULL CHECK (amount >= 0),
    currency CHAR(3) NOT NULL DEFAULT 'USD',
    payment_method TEXT NOT NULL,
    product_category TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('paid', 'refunded', 'failed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE sessions (
    session_id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_start_ts TIMESTAMPTZ NOT NULL,
    session_end_ts TIMESTAMPTZ NOT NULL,
    device_type TEXT NOT NULL,
    traffic_source TEXT NOT NULL,
    page_views INTEGER NOT NULL CHECK (page_views >= 1),
    actions_count INTEGER NOT NULL CHECK (actions_count >= 0),
    support_tickets_count INTEGER NOT NULL CHECK (support_tickets_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (session_end_ts >= session_start_ts)
);

CREATE INDEX idx_transactions_user_ts ON transactions (user_id, transaction_ts);
CREATE INDEX idx_transactions_status_ts ON transactions (status, transaction_ts);
CREATE INDEX idx_sessions_user_start ON sessions (user_id, session_start_ts);
CREATE INDEX idx_users_signup ON users (signup_ts);

INSERT INTO users (
    user_id,
    signup_ts,
    country,
    acquisition_channel,
    plan_type,
    marketing_opt_in,
    age,
    gender
)
SELECT
    gs AS user_id,
    timestamp '2024-01-01'
        + (random() * interval '520 days') AS signup_ts,
    CASE
        WHEN random() < 0.34 THEN 'US'
        WHEN random() < 0.55 THEN 'PL'
        WHEN random() < 0.70 THEN 'DE'
        WHEN random() < 0.84 THEN 'GB'
        WHEN random() < 0.93 THEN 'FR'
        ELSE 'OTHER'
    END AS country,
    CASE
        WHEN random() < 0.31 THEN 'organic'
        WHEN random() < 0.54 THEN 'paid_search'
        WHEN random() < 0.72 THEN 'social'
        WHEN random() < 0.87 THEN 'referral'
        ELSE 'partner'
    END AS acquisition_channel,
    CASE
        WHEN random() < 0.58 THEN 'free'
        WHEN random() < 0.84 THEN 'basic'
        WHEN random() < 0.96 THEN 'pro'
        ELSE 'enterprise'
    END AS plan_type,
    random() < 0.63 AS marketing_opt_in,
    LEAST(90, GREATEST(16, floor(18 + random() * 48 + random() * 18)::integer)) AS age,
    CASE
        WHEN random() < 0.48 THEN 'female'
        WHEN random() < 0.96 THEN 'male'
        ELSE 'other'
    END AS gender
FROM generate_series(1, 15000) AS gs;

INSERT INTO transactions (
    user_id,
    transaction_ts,
    amount,
    currency,
    payment_method,
    product_category,
    status
)
SELECT
    u.user_id,
    u.signup_ts + random() * GREATEST(now() - u.signup_ts, interval '1 day') AS transaction_ts,
    round((
        CASE u.plan_type
            WHEN 'free' THEN 5 + random() * 35
            WHEN 'basic' THEN 15 + random() * 80
            WHEN 'pro' THEN 40 + random() * 180
            ELSE 250 + random() * 900
        END
    )::numeric, 2) AS amount,
    'USD' AS currency,
    CASE
        WHEN random() < 0.47 THEN 'card'
        WHEN random() < 0.69 THEN 'paypal'
        WHEN random() < 0.86 THEN 'apple_pay'
        WHEN random() < 0.96 THEN 'bank_transfer'
        ELSE 'crypto'
    END AS payment_method,
    CASE
        WHEN random() < 0.42 THEN 'subscription'
        WHEN random() < 0.62 THEN 'addon'
        WHEN random() < 0.79 THEN 'usage_pack'
        WHEN random() < 0.92 THEN 'training'
        ELSE 'services'
    END AS product_category,
    CASE
        WHEN random() < 0.89 THEN 'paid'
        WHEN random() < 0.95 THEN 'failed'
        ELSE 'refunded'
    END AS status
FROM users AS u
CROSS JOIN LATERAL generate_series(
    1,
    CASE u.plan_type
        WHEN 'free' THEN 1 + floor(random() * 4)::integer
        WHEN 'basic' THEN 2 + floor(random() * 7)::integer
        WHEN 'pro' THEN 3 + floor(random() * 10)::integer
        ELSE 4 + floor(random() * 12)::integer
    END
) AS tx(n);

INSERT INTO sessions (
    user_id,
    session_start_ts,
    session_end_ts,
    device_type,
    traffic_source,
    page_views,
    actions_count,
    support_tickets_count
)
WITH generated_sessions AS (
    SELECT
        u.user_id,
        u.plan_type,
        u.signup_ts + random() * GREATEST(now() - u.signup_ts, interval '1 day') AS session_start_ts,
        CASE
            WHEN u.plan_type = 'enterprise' THEN 8 + floor(random() * 16)::integer
            WHEN u.plan_type = 'pro' THEN 5 + floor(random() * 14)::integer
            WHEN u.plan_type = 'basic' THEN 3 + floor(random() * 10)::integer
            ELSE 1 + floor(random() * 8)::integer
        END AS page_views
    FROM users AS u
    CROSS JOIN LATERAL generate_series(
        1,
        CASE u.plan_type
            WHEN 'free' THEN 1 + floor(random() * 4)::integer
            WHEN 'basic' THEN 2 + floor(random() * 6)::integer
            WHEN 'pro' THEN 3 + floor(random() * 8)::integer
            ELSE 4 + floor(random() * 10)::integer
        END
    ) AS ss(n)
)
SELECT
    user_id,
    session_start_ts,
    session_start_ts + (5 + floor(random() * 115)::integer) * interval '1 minute' AS session_end_ts,
    CASE
        WHEN random() < 0.52 THEN 'mobile'
        WHEN random() < 0.86 THEN 'desktop'
        ELSE 'tablet'
    END AS device_type,
    CASE
        WHEN random() < 0.35 THEN 'direct'
        WHEN random() < 0.58 THEN 'search'
        WHEN random() < 0.75 THEN 'email'
        WHEN random() < 0.90 THEN 'social'
        ELSE 'ads'
    END AS traffic_source,
    page_views,
    GREATEST(0, page_views + floor(random() * 8)::integer - 2) AS actions_count,
    CASE
        WHEN random() < 0.89 THEN 0
        WHEN random() < 0.98 THEN 1
        ELSE 2
    END AS support_tickets_count
FROM generated_sessions;

ANALYZE users;
ANALYZE transactions;
ANALYZE sessions;
