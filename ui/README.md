# Remember Twelve - Web UI

Beautiful Material Design web interface to view your curated Twelve.

## Features

- **Material Design 3** - Clean, modern Google Material Design
- **Responsive Grid** - Beautiful photo cards that adapt to any screen
- **Score Visualization** - Color-coded badges for quality, emotional, and combined scores
- **Monthly Timeline** - Visual representation of temporal distribution
- **Photo Details** - Click any photo for full view with metadata
- **Real-time Updates** - Instant rendering of your curated selections

## Quick Start

```bash
# Run the web viewer
python ui/twelve_viewer.py twelve_2025_balanced.json

# Custom port
python ui/twelve_viewer.py twelve_2025_balanced.json --port 8080

# Don't auto-open browser
python ui/twelve_viewer.py twelve_2025_balanced.json --no-browser
```

The UI will automatically open in your default browser at `http://localhost:5000`

## UI Components

### Gallery View
- **Header** - Year title with statistics
- **Timeline** - Monthly distribution pills (active months highlighted)
- **Photo Grid** - Responsive 3-column grid (mobile: 1 column)
- **Photo Cards** - Each showing:
  - Rank badge (#1-12)
  - Photo preview
  - Month and date
  - Three score badges (Combined, Quality, Emotional)
  - Color coding (green=high, orange=medium, red=low)

### Photo Detail Modal
- Full-size photo view
- Complete metadata
- All scores displayed
- Face detection status

## Design System

### Colors
- **Primary**: Purple gradient (#667eea → #764ba2)
- **Success**: Green (#2e7d32)
- **Warning**: Orange (#e65100)
- **Error**: Red (#c62828)

### Typography
- **Font**: Roboto (Google's Material Design font)
- **Weights**: 300 (light), 400 (regular), 500 (medium), 700 (bold)

### Components
- **Cards**: 16px border radius, elevation 4dp
- **Buttons**: Material ripple effects
- **Icons**: Material Icons font
- **Modals**: Materialize CSS modals

## File Structure

```
ui/
├── twelve_viewer.py          # Flask web server
├── templates/
│   ├── gallery.html          # Main gallery view
│   └── error.html            # Error page
├── static/
│   ├── css/                  # Custom CSS (if needed)
│   ├── js/                   # Custom JS (if needed)
│   └── images/               # Static images
└── README.md                 # This file
```

## Tech Stack

- **Backend**: Flask (Python web framework)
- **Frontend**: HTML5, CSS3, JavaScript
- **UI Framework**: Materialize CSS 1.0.0
- **Icons**: Material Icons
- **Fonts**: Google Fonts (Roboto)

## Customization

### Change Colors

Edit `gallery.html` and modify the CSS variables:

```css
/* Primary gradient */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Score badge colors */
.score-badge.high { background: #e8f5e9; color: #2e7d32; }
```

### Add Custom Features

The Flask app is simple to extend:

```python
@app.route('/export')
def export_selection():
    # Add CSV export
    # Add print view
    # Add sharing features
```

## Browser Support

- ✅ Chrome/Edge (latest)
- ✅ Firefox (latest)
- ✅ Safari (latest)
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

## Screenshots

### Gallery View
- Grid of 12 photos with scores
- Monthly timeline at top
- Statistics in header

### Detail Modal
- Full-size photo
- Complete metadata
- Click anywhere outside to close

## Future Enhancements

- [ ] Export to PDF
- [ ] Share to social media
- [ ] Print-friendly view
- [ ] Comparison view (multiple years side-by-side)
- [ ] Slideshow mode
- [ ] Filtering by month/score
- [ ] Search functionality

## Troubleshooting

**Photos not loading?**
- Check that photo paths in JSON are correct
- Ensure photos directory is accessible

**Port already in use?**
- Use `--port` flag to specify different port
- Kill existing Flask processes

**Browser doesn't open?**
- Use `--no-browser` flag and open manually
- Navigate to `http://localhost:5000`

## Development

Run in debug mode (auto-reload):

```bash
export FLASK_ENV=development
python ui/twelve_viewer.py twelve_2025_balanced.json
```

## License

Part of Remember Twelve project.

---

**Remember Twelve**: Preserve your year in twelve unforgettable moments.
