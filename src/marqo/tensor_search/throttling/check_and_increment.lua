local set_key = KEYS[1]

local thread_name = ARGV[1]
local thread_limit = tonumber(ARGV[2])
local expire_time = tonumber(ARGV[3])

-- Get expiry time
local now = redis.call('time')[1]
local expiry_threshold = now - expire_time

-- Expire items in sorted set past the threshhold
redis.call('zremrangebyscore', set_key, '-inf', expiry_threshold)

-- Current count is items in sorted set
local current_thread_count = redis.call('zcard', set_key)

if current_thread_count + 1 > thread_limit then
    -- If over concurrency limit, return 0 for failure
    return 0
else
    -- add item to sorted set (time, key name)
    redis.call("zadd", set_key, now, thread_name)

    --DEBUG
    local debug_msg = "DEBUG REDIS: "
    debug_msg = debug_msg .. "expiry_threshold: " .. expiry_threshold .. " | "
    debug_msg = debug_msg .. "now: " .. now .. " | "
    debug_msg = debug_msg .. "expire_time: " .. expire_time .. " | "
    debug_msg = debug_msg .. "set_key: " .. set_key .. " | "
    debug_msg = debug_msg .. "current_thread_count: " .. current_thread_count .. " | "
    debug_msg = debug_msg .. "thread_limit: " .. thread_limit .. " | "
    return debug_msg
    
    -- return: expiry_threshold, now, expire_time, set_key, current_thread_count, thread_limit, 
    -- return thread_name
end