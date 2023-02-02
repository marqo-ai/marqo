local set_key = KEYS[1]

local thread_name = ARGV[1]
local thread_limit = tonumber(ARGV[2])
local expire_time = ARGV[3]

-- Get expiry time
local now = redis.call('time')[1]
local expiry_threshold = now - expire_time

-- Expire items in sorted set past the threshhold
redis.call('zremrangebyscore', set_key, '-inf', timestamp_threshold)

-- Current count is items in sorted set
local current_thread_count = redis.call('zcard', set_key)

if current_thread_count + 1 > thread_limit then
    -- If over concurrency limit, return 0 for failure
    return 0
else
    -- add item to sorted set (time, key name)
    redis.call("zadd", set_key, now, thread_name)
    return thread_key
end