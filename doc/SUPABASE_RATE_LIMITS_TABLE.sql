-- Supabase Rate Limits Table
-- Create this table in your Supabase project SQL editor
-- Required for user-based rate limiting with Supabase storage

CREATE TABLE IF NOT EXISTS rate_limits (
    user_id TEXT PRIMARY KEY,
    request_count INTEGER DEFAULT 0,
    window_start DOUBLE PRECISION NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for faster cleanup queries
CREATE INDEX IF NOT EXISTS idx_rate_limits_window_start ON rate_limits(window_start);

-- Enable Row Level Security (RLS)
ALTER TABLE rate_limits ENABLE ROW LEVEL SECURITY;

-- Policy: Allow API server to read/write (using anon key)
-- Note: In production, use a service role key or create specific policies
CREATE POLICY "Allow anon insert" ON rate_limits
    FOR INSERT TO anon
    WITH CHECK (true);

CREATE POLICY "Allow anon select" ON rate_limits
    FOR SELECT TO anon
    USING (true);

CREATE POLICY "Allow anon update" ON rate_limits
    FOR UPDATE TO anon
    USING (true)
    WITH CHECK (true);

-- Optional: Function to cleanup expired entries
CREATE OR REPLACE FUNCTION cleanup_rate_limits(window_seconds INTEGER DEFAULT 3600)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM rate_limits
    WHERE window_start < EXTRACT(EPOCH FROM NOW()) - window_seconds;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Optional: Scheduled cleanup (requires pg_cron extension enabled in Supabase)
-- SELECT cron.schedule('cleanup-rate-limits', '*/15 * * * *', $$SELECT cleanup_rate_limits(3600)$$);

COMMENT ON TABLE rate_limits IS 'JWT user-based rate limiting counters';
COMMENT ON COLUMN rate_limits.user_id IS 'User ID from JWT sub claim';
COMMENT ON COLUMN rate_limits.request_count IS 'Number of requests in current window';
COMMENT ON COLUMN rate_limits.window_start IS 'Unix timestamp of window start';
COMMENT ON COLUMN rate_limits.updated_at IS 'Last update timestamp';
