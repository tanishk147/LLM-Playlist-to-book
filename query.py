#!/usr/bin/env python3
import sys
import sqlite3
import json

def format_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m{secs}s"

def main():
    if len(sys.argv) < 2:
        print("Usage: python query.py <claim_id>")
        print("Example: python query.py c2914  (or just 2914)")
        sys.exit(1)
        
    raw_arg = sys.argv[1]
    claim_id_str = raw_arg.strip().lower().lstrip('c')
    try:
        claim_id = int(claim_id_str)
    except ValueError:
        print(f"Error: Invalid claim ID format '{raw_arg}'. Use e.g. c2914 or 2914.")
        sys.exit(1)
        
    db_path = "data/claims.sqlite"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    row = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
    conn.close()
    
    if not row:
        print(f"Claim c{claim_id} not found in {db_path}.")
        sys.exit(1)
        
    video_id = row['video_id']
    ts_start = row['ts_start']
    ts_end = row['ts_end']
    content = row['content']
    
    # Standard YouTube video + timestamp URL format
    t_seconds = int(ts_start)
    youtube_url = f"https://www.youtube.com/watch?v={video_id}&t={t_seconds}s"
    
    print("\n" + "="*80)
    print(f"🔍 CLAIM c{claim_id} DETAILS")
    print("="*80)
    print(f"📝 Content:   {content}")
    print(f"📺 Video ID:  {video_id}")
    print(f"⏰ Timestamps: {format_time(ts_start)} to {format_time(ts_end)} (seconds: {ts_start} - {ts_end})")
    print(f"🔗 YouTube:   {youtube_url}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
