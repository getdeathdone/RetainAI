-- RetainAI: feature mart extraction script.
-- Builds one row per user with RFM, engagement, sequence, trend and churn-label features.

SET search_path TO retainai;

DROP MATERIALIZED VIEW IF EXISTS ml_customer_features;

CREATE MATERIALIZED VIEW ml_customer_features AS
WITH params AS (
    SELECT now()::timestamptz AS as_of_ts
),
paid_transactions AS (
    SELECT
        t.*,
        row_number() OVER (
            PARTITION BY t.user_id
            ORDER BY t.transaction_ts
        ) AS tx_number,
        row_number() OVER (
            PARTITION BY t.user_id
            ORDER BY t.transaction_ts DESC
        ) AS recent_tx_rank,
        lag(t.transaction_ts) OVER (
            PARTITION BY t.user_id
            ORDER BY t.transaction_ts
        ) AS prev_transaction_ts,
        sum(t.amount) OVER (
            PARTITION BY t.user_id
            ORDER BY t.transaction_ts
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_revenue,
        avg(t.amount) OVER (
            PARTITION BY t.user_id
            ORDER BY t.transaction_ts
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS rolling_3tx_avg_amount
    FROM transactions AS t
    WHERE t.status = 'paid'
),
transaction_features AS (
    SELECT
        pt.user_id,
        count(*) AS paid_tx_count,
        sum(pt.amount) AS monetary_value,
        avg(pt.amount) AS avg_order_value,
        max(pt.amount) AS max_order_value,
        min(pt.amount) AS min_order_value,
        max(pt.transaction_ts) AS last_paid_tx_ts,
        min(pt.transaction_ts) AS first_paid_tx_ts,
        avg(EXTRACT(epoch FROM (pt.transaction_ts - pt.prev_transaction_ts)) / 86400.0)
            FILTER (WHERE pt.prev_transaction_ts IS NOT NULL) AS avg_days_between_paid_tx,
        max(pt.cumulative_revenue) AS cumulative_revenue,
        max(pt.rolling_3tx_avg_amount) FILTER (WHERE pt.recent_tx_rank = 1)
            AS last_rolling_3tx_avg_amount
    FROM paid_transactions AS pt
    GROUP BY pt.user_id
),
transaction_status_features AS (
    SELECT
        t.user_id,
        count(*) FILTER (WHERE t.status = 'failed') AS failed_tx_count,
        count(*) FILTER (WHERE t.status = 'refunded') AS refunded_tx_count,
        count(*) AS all_tx_count,
        count(DISTINCT t.product_category) AS product_category_count,
        count(DISTINCT t.payment_method) AS payment_method_count
    FROM transactions AS t
    GROUP BY t.user_id
),
session_windows AS (
    SELECT
        s.*,
        lag(s.session_start_ts) OVER (
            PARTITION BY s.user_id
            ORDER BY s.session_start_ts
        ) AS prev_session_start_ts,
        row_number() OVER (
            PARTITION BY s.user_id
            ORDER BY s.session_start_ts DESC
        ) AS recent_session_rank,
        avg(s.actions_count) OVER (
            PARTITION BY s.user_id
            ORDER BY s.session_start_ts
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) AS rolling_5session_actions,
        avg(s.page_views) OVER (
            PARTITION BY s.user_id
            ORDER BY s.session_start_ts
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) AS rolling_5session_page_views
    FROM sessions AS s
),
session_features AS (
    SELECT
        sw.user_id,
        count(*) AS session_count,
        max(sw.session_start_ts) AS last_session_ts,
        min(sw.session_start_ts) AS first_session_ts,
        avg(EXTRACT(epoch FROM (sw.session_end_ts - sw.session_start_ts)) / 60.0) AS avg_session_minutes,
        sum(sw.page_views) AS total_page_views,
        avg(sw.page_views) AS avg_page_views,
        sum(sw.actions_count) AS total_actions,
        avg(sw.actions_count) AS avg_actions,
        sum(sw.support_tickets_count) AS support_tickets_count,
        avg(EXTRACT(epoch FROM (sw.session_start_ts - sw.prev_session_start_ts)) / 86400.0)
            FILTER (WHERE sw.prev_session_start_ts IS NOT NULL) AS avg_days_between_sessions,
        avg(sw.actions_count) FILTER (WHERE sw.recent_session_rank <= 3) AS recent_3_sessions_avg_actions,
        avg(sw.page_views) FILTER (WHERE sw.recent_session_rank <= 3) AS recent_3_sessions_avg_page_views,
        max(sw.rolling_5session_actions) FILTER (WHERE sw.recent_session_rank = 1) AS last_rolling_5session_actions,
        max(sw.rolling_5session_page_views) FILTER (WHERE sw.recent_session_rank = 1) AS last_rolling_5session_page_views
    FROM session_windows AS sw
    GROUP BY sw.user_id
),
monthly_activity AS (
    SELECT
        u.user_id,
        date_trunc('month', activity_ts)::date AS activity_month,
        count(*) AS activity_events
    FROM users AS u
    LEFT JOIN (
        SELECT user_id, transaction_ts AS activity_ts
        FROM transactions
        WHERE status = 'paid'
        UNION ALL
        SELECT user_id, session_start_ts AS activity_ts
        FROM sessions
    ) AS activity ON activity.user_id = u.user_id
    GROUP BY u.user_id, date_trunc('month', activity_ts)::date
),
activity_trends AS (
    SELECT
        ma.user_id,
        avg(ma.activity_events) FILTER (
            WHERE ma.activity_month >= date_trunc('month', (SELECT as_of_ts FROM params))::date - interval '2 months'
        ) AS avg_monthly_events_last_3m,
        avg(ma.activity_events) FILTER (
            WHERE ma.activity_month < date_trunc('month', (SELECT as_of_ts FROM params))::date - interval '2 months'
        ) AS avg_monthly_events_before_3m,
        regr_slope(
            ma.activity_events::double precision,
            EXTRACT(epoch FROM ma.activity_month::timestamp)::double precision
        ) AS monthly_activity_slope
    FROM monthly_activity AS ma
    WHERE ma.activity_month IS NOT NULL
    GROUP BY ma.user_id
),
last_event AS (
    SELECT
        user_id,
        max(event_ts) AS last_event_ts
    FROM (
        SELECT user_id, transaction_ts AS event_ts
        FROM transactions
        WHERE status = 'paid'
        UNION ALL
        SELECT user_id, session_start_ts AS event_ts
        FROM sessions
    ) AS events
    GROUP BY user_id
)
SELECT
    u.user_id,
    u.signup_ts,
    u.country,
    u.acquisition_channel,
    u.plan_type,
    u.marketing_opt_in,
    u.age,
    u.gender,
    EXTRACT(day FROM ((SELECT as_of_ts FROM params) - u.signup_ts))::integer AS account_age_days,
    EXTRACT(day FROM ((SELECT as_of_ts FROM params) - COALESCE(le.last_event_ts, u.signup_ts)))::integer AS recency_days,
    EXTRACT(day FROM ((SELECT as_of_ts FROM params) - COALESCE(tf.last_paid_tx_ts, u.signup_ts)))::integer AS payment_recency_days,
    COALESCE(tf.paid_tx_count, 0) AS paid_tx_count,
    COALESCE(tsf.all_tx_count, 0) AS all_tx_count,
    COALESCE(tsf.failed_tx_count, 0) AS failed_tx_count,
    COALESCE(tsf.refunded_tx_count, 0) AS refunded_tx_count,
    COALESCE(tsf.product_category_count, 0) AS product_category_count,
    COALESCE(tsf.payment_method_count, 0) AS payment_method_count,
    COALESCE(tf.monetary_value, 0)::numeric(12, 2) AS monetary_value,
    COALESCE(tf.avg_order_value, 0)::numeric(10, 2) AS avg_order_value,
    COALESCE(tf.max_order_value, 0)::numeric(10, 2) AS max_order_value,
    COALESCE(tf.min_order_value, 0)::numeric(10, 2) AS min_order_value,
    COALESCE(tf.avg_days_between_paid_tx, 0)::numeric(10, 2) AS avg_days_between_paid_tx,
    COALESCE(tf.cumulative_revenue, 0)::numeric(12, 2) AS ltv_observed,
    COALESCE(tf.last_rolling_3tx_avg_amount, 0)::numeric(10, 2) AS last_rolling_3tx_avg_amount,
    COALESCE(sf.session_count, 0) AS session_count,
    COALESCE(sf.total_page_views, 0) AS total_page_views,
    COALESCE(sf.avg_page_views, 0)::numeric(10, 2) AS avg_page_views,
    COALESCE(sf.total_actions, 0) AS total_actions,
    COALESCE(sf.avg_actions, 0)::numeric(10, 2) AS avg_actions,
    COALESCE(sf.avg_session_minutes, 0)::numeric(10, 2) AS avg_session_minutes,
    COALESCE(sf.support_tickets_count, 0) AS support_tickets_count,
    COALESCE(sf.avg_days_between_sessions, 0)::numeric(10, 2) AS avg_days_between_sessions,
    COALESCE(sf.recent_3_sessions_avg_actions, 0)::numeric(10, 2) AS recent_3_sessions_avg_actions,
    COALESCE(sf.recent_3_sessions_avg_page_views, 0)::numeric(10, 2) AS recent_3_sessions_avg_page_views,
    COALESCE(sf.last_rolling_5session_actions, 0)::numeric(10, 2) AS last_rolling_5session_actions,
    COALESCE(sf.last_rolling_5session_page_views, 0)::numeric(10, 2) AS last_rolling_5session_page_views,
    COALESCE(at.avg_monthly_events_last_3m, 0)::numeric(10, 2) AS avg_monthly_events_last_3m,
    COALESCE(at.avg_monthly_events_before_3m, 0)::numeric(10, 2) AS avg_monthly_events_before_3m,
    COALESCE(at.monthly_activity_slope, 0)::numeric(18, 8) AS monthly_activity_slope,
    CASE
        WHEN le.last_event_ts IS NULL THEN 1
        WHEN EXTRACT(day FROM ((SELECT as_of_ts FROM params) - le.last_event_ts)) >= 45 THEN 1
        WHEN u.plan_type = 'free'
            AND EXTRACT(day FROM ((SELECT as_of_ts FROM params) - le.last_event_ts)) >= 30 THEN 1
        ELSE 0
    END AS churn_label_45d,
    CASE
        WHEN COALESCE(tf.cumulative_revenue, 0) >= 1000 THEN 'high'
        WHEN COALESCE(tf.cumulative_revenue, 0) >= 250 THEN 'medium'
        ELSE 'low'
    END AS ltv_segment,
    (SELECT as_of_ts FROM params) AS extracted_at
FROM users AS u
LEFT JOIN transaction_features AS tf ON tf.user_id = u.user_id
LEFT JOIN transaction_status_features AS tsf ON tsf.user_id = u.user_id
LEFT JOIN session_features AS sf ON sf.user_id = u.user_id
LEFT JOIN activity_trends AS at ON at.user_id = u.user_id
LEFT JOIN last_event AS le ON le.user_id = u.user_id;

CREATE UNIQUE INDEX idx_ml_customer_features_user_id
    ON ml_customer_features (user_id);

CREATE INDEX idx_ml_customer_features_churn
    ON ml_customer_features (churn_label_45d);

CREATE INDEX idx_ml_customer_features_ltv_segment
    ON ml_customer_features (ltv_segment);
