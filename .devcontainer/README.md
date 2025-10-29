# Remember Twelve - DevContainer Configuration

This devcontainer configuration enables you to develop Remember Twelve in GitHub Codespaces with full access to Amplifier agents and tools.

## What's Included

### Base Environment
- **Python 3.11** - Core application runtime
- **Node.js 20** - For any frontend tooling
- **Git** - Version control
- **GitHub CLI** - For GitHub operations

### Amplifier Integration
- **Auto-cloned** from https://github.com/cpark4x/amplifier
- **Symlinked** to `./amplifier` for easy access
- **Full agent access** - /ultrathink-task, zen-architect, modular-builder, etc.

### VS Code Extensions
- Python + Pylance (IntelliSense)
- Ruff (Linting & Formatting)
- Prettier (Code formatting)
- ESLint (JavaScript linting)

### Application Setup
- Database initialized at `~/.remember_twelve/remember_twelve.db`
- Photo directories created
- All Python dependencies installed
- Ready to run immediately

## How to Use

### Option 1: GitHub Codespaces (Recommended)

1. **Open in Codespace:**
   ```bash
   gh codespace create -r cpark4x/remember-twelve
   ```

2. **Or via GitHub UI:**
   - Go to repository
   - Click "Code" → "Codespaces" → "Create codespace on main"

3. **Wait for setup** (2-3 minutes for post-create script)

4. **Start the app:**
   ```bash
   python remember_twelve_app.py start
   ```

5. **Access via forwarded port** (Codespace will prompt)

### Option 2: Local Dev Container

1. **Open in VS Code:**
   - Open the remember-twelve folder
   - Press `F1`
   - Select "Dev Containers: Reopen in Container"

2. **Wait for build** (first time takes longer)

3. **Start developing!**

## Using Amplifier in Codespace

All Amplifier agents are available via Claude Code:

```bash
# General task orchestration
/ultrathink-task "Add export to PDF feature"

# Architecture review
/ddd:status

# Direct agent use (if configured)
# zen-architect for design decisions
# modular-builder for implementation
# bug-hunter for debugging
```

### Where is Amplifier?

- **Cloned to:** `~/toolkits/amplifier`
- **Symlinked to:** `./amplifier` (in project root)
- **Agents available:** All agents from Amplifier repo

## Port Forwarding

The devcontainer automatically forwards these ports:

| Port | Purpose | Visibility |
|------|---------|------------|
| 8000 | Main application server | Public (with notification) |
| 8001 | Development server | Private |

Codespaces will give you a URL like:
`https://scaling-chainsaw-abc123.github.dev`

## Environment Variables

- `REMEMBER_TWELVE_ENV=codespace` - Indicates running in Codespace
- Use this to conditionally adjust behavior (e.g., URLs, file paths)

## Post-Create Script

The `.devcontainer/post-create.sh` script runs automatically and:

1. ✅ Installs Python dependencies
2. ✅ Installs Node dependencies (if package.json exists)
3. ✅ Clones Amplifier toolkit
4. ✅ Creates symlink to Amplifier
5. ✅ Sets up `~/.remember_twelve/` directories
6. ✅ Initializes SQLite database
7. ✅ Configures Claude Code

**Total setup time:** ~2-3 minutes

## Customization

### Add More Tools

Edit `.devcontainer/devcontainer.json`:

```json
"features": {
  "ghcr.io/devcontainers/features/docker-in-docker:1": {}
}
```

### Modify Post-Create Steps

Edit `.devcontainer/post-create.sh`:

```bash
# Add your custom setup steps
echo "Installing additional tools..."
pip install <your-package>
```

### Change Python Version

Edit the image in `devcontainer.json`:

```json
"image": "mcr.microsoft.com/devcontainers/python:1-3.12-bookworm"
```

## Troubleshooting

### Amplifier not found
```bash
# Manually clone if post-create failed
git clone https://github.com/cpark4x/amplifier.git ~/toolkits/amplifier
ln -s ~/toolkits/amplifier amplifier
```

### Database not initialized
```bash
python -c "from src.database import init_db; init_db()"
```

### Ports not forwarding
- Check Codespace port forwarding settings
- Make sure app is running on 0.0.0.0, not localhost
- Use VS Code ports tab to manually forward

### Dependencies not installed
```bash
pip install -r requirements.txt
```

## Differences from Local Development

### File Paths
- Local: `~/amplifier/remember-twelve`
- Codespace: `/workspaces/remember-twelve`

### Database Location
- Same: `~/.remember_twelve/remember_twelve.db`
- Persists across Codespace rebuilds

### Amplifier Location
- Local: `~/dev/toolkits/amplifier`
- Codespace: `~/toolkits/amplifier`

### Network Access
- Local: `http://localhost:8000`
- Codespace: `https://<random-name>.github.dev` (HTTPS with auth)

## Performance Considerations

### Codespace Resources
Default Codespace:
- 2 cores
- 8 GB RAM
- 32 GB disk

Upgrade if needed for heavy workloads.

### Rebuild vs. Restart
- **Rebuild:** Full container rebuild (slow, needed after devcontainer.json changes)
- **Restart:** Fast restart (keeps environment)

## Security Notes

### Secrets Management
- Never commit `.env` files
- Use Codespace secrets for sensitive data
- Google Photos tokens stored in `~/.remember_twelve/tokens.db`

### Public Ports
- Port 8000 is public by default
- Change to private in port settings if needed
- Add authentication for production use

## Next Steps

1. **Test the Codespace** - Create one and verify setup
2. **Migrate data** - Copy 2023 photos if needed
3. **Use Amplifier** - Try /ultrathink-task commands
4. **Develop features** - Everything works like local!

## Benefits of This Setup

✅ **Consistent environment** - Same setup every time
✅ **Cloud development** - Work from any device
✅ **Amplifier integration** - Full agent access
✅ **Pre-configured tools** - No manual setup
✅ **Fast onboarding** - 2-3 minutes to productive
✅ **Shareable** - Team members get identical setups

---

*Built with devcontainer-templates from https://github.com/cpark4x/devcontainer-templates*
