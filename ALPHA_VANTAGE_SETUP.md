# Alpha Vantage API Setup

## Why Alpha Vantage?

Yahoo Finance has strict limits on historical intraday data:
- 1-minute data: Only last **60 days**
- 5-minute data: Only last **60 days**
- 15-minute data: Only last **60 days**
- 1-hour data: Only last **730 days** (2 years)

**Alpha Vantage** provides better coverage:
- 1-minute, 5-minute, 15-minute, 30-minute, 60-minute forex data
- **Free tier**: Last ~1-2 months of intraday data
- **Premium tier**: Up to 20+ years of intraday data

## Get Your FREE API Key (30 seconds)

1. **Go to**: https://www.alphavantage.co/support/#api-key

2. **Enter your email** and click "GET FREE API KEY"

3. **Copy the API key** (looks like: `ABCD1234EFGH5678`)

4. **Add it to config.yaml**:

```bash
# Open config file
notepad config\config.yaml

# Find this line:
alphavantage_api_key: "YOUR_API_KEY_HERE"

# Replace with your actual key:
alphavantage_api_key: "ABCD1234EFGH5678"

# Save and close
```

## Free Tier Limits

- **25 API calls per day**
- **5 API calls per minute**

For this project, downloading all timeframes (1m, 5m, 15m, 1h) = **4 API calls**

So you can download data for **6 different days** per day (24 calls).

The data collector script automatically waits 12 seconds between requests to respect the rate limit.

## What You Get (Free Tier)

- EUR/USD 1-minute data: ~30-60 days
- EUR/USD 5-minute data: ~30-60 days
- EUR/USD 15-minute data: ~30-60 days
- EUR/USD 1-hour data: ~30-60 days

**This is MUCH better than Yahoo's 60-day limit for intraday!**

## Premium Tier (Optional)

If you need more data or higher API limits:

- **$49.99/month**: 75 calls/minute, 20 years of data
- **$149.99/month**: 150 calls/minute, 20 years of data
- **$499.99/month**: 600 calls/minute, 20 years of data

See: https://www.alphavantage.co/premium/

## Usage

Once you've added your API key to `config.yaml`:

```bash
# Collect data (will use Alpha Vantage automatically)
python collect_data.py --use-ohlc-only
```

The script will:
1. Try Alpha Vantage first (better coverage)
2. Fallback to Yahoo Finance if Alpha Vantage fails
3. Automatically handle rate limiting (12 sec between calls)

## Troubleshooting

### "API key required"
- Make sure you added the key to `config/config.yaml`
- Make sure it's not still "YOUR_API_KEY_HERE"

### "API limit reached"
- Free tier: 25 calls/day, 5 calls/minute
- Wait until tomorrow or upgrade to premium

### "No data returned"
- Free tier has limited historical data (~1-2 months)
- Consider Dukascopy tick data for longer history: `python collect_data.py --use-tick-data`

## Alternative: Use Dukascopy Tick Data

If you need more than 1-2 months of data:

```bash
# Download 3 years of tick data (takes 2-4 hours)
python collect_data.py --use-tick-data
```

This downloads from Dukascopy (free, no API key needed, full historical data).

---

**Get your free API key now**: https://www.alphavantage.co/support/#api-key
