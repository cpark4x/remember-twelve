#!/bin/bash
set -e

echo "==========================================="
echo "Remember Twelve - Codespace Setup"
echo "==========================================="
echo

# ============================================
# 1. Install Python dependencies
# ============================================
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Python dependencies installed"
echo

# ============================================
# 2. Install Node dependencies (if needed)
# ============================================
if [ -f "package.json" ]; then
    echo "📦 Installing Node dependencies..."
    npm install
    echo "✓ Node dependencies installed"
    echo
fi

# ============================================
# 3. Clone and setup Amplifier
# ============================================
echo "🔧 Setting up Amplifier toolkit..."

# Create toolkits directory
mkdir -p ~/toolkits

# Check if Amplifier already exists
if [ ! -d ~/toolkits/amplifier ]; then
    echo "  Cloning Amplifier from GitHub..."
    git clone https://github.com/cpark4x/amplifier.git ~/toolkits/amplifier
    echo "  ✓ Amplifier cloned"
else
    echo "  ✓ Amplifier already exists, pulling latest..."
    cd ~/toolkits/amplifier
    git pull
    cd -
fi

# Create symlink if it doesn't exist
if [ ! -L "amplifier" ]; then
    echo "  Creating amplifier symlink..."
    ln -s ~/toolkits/amplifier amplifier
    echo "  ✓ Symlink created"
else
    echo "  ✓ Amplifier symlink already exists"
fi

# Create .claude directory and symlink commands/agents/tools/settings
echo "  Setting up Claude Code integration..."
mkdir -p .claude
if [ ! -L ".claude/commands" ]; then
    ln -s ../amplifier/.claude/commands .claude/commands
    echo "  ✓ Commands symlinked"
fi
if [ ! -L ".claude/agents" ]; then
    ln -s ../amplifier/.claude/agents .claude/agents
    echo "  ✓ Agents symlinked"
fi
if [ ! -L ".claude/tools" ]; then
    ln -s ../amplifier/.claude/tools .claude/tools
    echo "  ✓ Tools symlinked"
fi
if [ ! -L ".claude/settings.json" ]; then
    ln -s ../amplifier/.claude/settings.json .claude/settings.json
    echo "  ✓ Settings symlinked"
fi

echo "✓ Amplifier setup complete"
echo

# ============================================
# 4. Setup Remember Twelve directories
# ============================================
echo "📁 Setting up Remember Twelve directories..."

# Create user data directory
mkdir -p ~/.remember_twelve/photos
mkdir -p ~/.remember_twelve/exports

echo "✓ Directories created"
echo

# ============================================
# 5. Initialize database (if not exists)
# ============================================
if [ ! -f ~/.remember_twelve/remember_twelve.db ]; then
    echo "🗄️  Initializing database..."
    python3 -c "from src.database import init_db; init_db()"
    echo "✓ Database initialized"
    echo
else
    echo "✓ Database already exists"
    echo
fi

# ============================================
# 6. Setup Claude Code configuration
# ============================================
echo "⚙️  Setting up Claude Code..."

# Create .claude directory if it doesn't exist
mkdir -p .claude

# Copy CLAUDE.md if it exists
if [ -f "CLAUDE.md" ]; then
    echo "  ✓ CLAUDE.md found"
else
    echo "  ⚠️  No CLAUDE.md found - create one to customize Claude Code behavior"
fi

echo "✓ Claude Code configuration ready"
echo

# ============================================
# 7. Display helpful information
# ============================================
echo "==========================================="
echo "✅ Setup Complete!"
echo "==========================================="
echo
echo "🚀 To start the app:"
echo "   python remember_twelve_app.py start"
echo
echo "🧠 To use Amplifier agents:"
echo "   /ultrathink-task <description>"
echo "   Available agents: zen-architect, modular-builder, bug-hunter, etc."
echo
echo "📚 Documentation:"
echo "   - README.md - Project overview"
echo "   - TRANSFORMATION_SUMMARY.md - Architecture details"
echo "   - V1_INTEGRATION_COMPLETE.md - Current status"
echo
echo "🔗 Amplifier location:"
echo "   ~/toolkits/amplifier (cloned)"
echo "   ./amplifier (symlinked)"
echo
echo "Happy coding! 🎉"
echo
