# Quick Reference: Makefile Commands

## JSON Export Commands

### Export game2.flop (recommended)
```bash
make export-json-game2
```
**Output:** `game2-complete.json` (330 MB) with sample data display

---

### Export any .flop file
```bash
make export-json FLOP_FILE=yourfile.flop OUTPUT_JSON=output.json
```

**Examples:**
```bash
# Default (game.flop → game-export.json)
make export-json

# Custom file
make export-json FLOP_FILE=game2.flop OUTPUT_JSON=complete.json
```

---

### Preview in console
```bash
make export-json-console FLOP_FILE=game2.flop
```
**Output:** Shows structure and sample data without creating large file

---

## What's Included in JSON Export?

✅ **Combos** - Hand combinations `[card1, card2]`
✅ **Equity** - Win probability for each hand (0-1)
✅ **EV** - Expected value in chips
✅ **EQR** - Equity Realization percentage
✅ **Strategies** - All node strategies
✅ **CFValues** - Counterfactual values
✅ **Game Tree** - Complete decision tree

---

## Example Output

```json
{
  "hand_data": {
    "oop_private_cards": [[40, 41], [40, 42], ...],
    "oop_equity": [0.6089, 0.6089, ...],
    "oop_ev": [36.67, 36.67, ...],
    "oop_eqr": [60.21, 60.21, ...],
    "ip_private_cards": [[0, 1], [0, 3], ...],
    "ip_equity": [0.8416, 0.8416, ...],
    "ip_ev": [46.87, 46.85, ...],
    "ip_eqr": [55.69, 55.67, ...]
  }
}
```

---

## Other Commands

### Start solver
```bash
make start CONFIG=config/template.json
```

### Start with debug logging
```bash
make start-debug CONFIG=config/template.json
```

### Build solver
```bash
make build
```

### Clean build
```bash
make clean
```

---

## Analyzing JSON Output

### Show hand counts
```bash
jq '{oop: (.hand_data.oop_private_cards | length), ip: (.hand_data.ip_private_cards | length)}' output.json
```

### Show first 5 hands with stats
```bash
jq '[range(0;5)] as $i | {
  hand: $i,
  combo: .hand_data.oop_private_cards[$i],
  equity: .hand_data.oop_equity[$i],
  ev: .hand_data.oop_ev[$i],
  eqr: .hand_data.oop_eqr[$i]
}' output.json
```

### Calculate averages
```bash
jq '{
  avg_oop_equity: (.hand_data.oop_equity | add / length),
  avg_oop_ev: (.hand_data.oop_ev | add / length)
}' output.json
```

---

## File Sizes

| File | Size | Description |
|------|------|-------------|
| game.flop | 9 MB | Binary format |
| game.flop → JSON | 114 MB | With all stats |
| game2.flop | 26 MB | Binary format |
| game2.flop → JSON | 330 MB | With all stats |

**Tip:** JSON files are 10-15x larger than binary files

---

## Performance

- **Load:** ~0.04s
- **Export:** ~1s
- **Write:** ~2 minutes (for 330 MB)
- **Total:** ~2-3 minutes

---

## Need Help?

See detailed documentation: `MAKEFILE_JSON_EXPORT_GUIDE.md`
