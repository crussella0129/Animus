# Animus VSCode Extension

AI coding assistant integrated directly into your IDE.

## Features

- **Chat Interface**: Conversational AI assistant in your sidebar
- **Context-Aware**: Automatically includes your current file and selection
- **Streaming Responses**: See AI responses as they generate
- **Code Actions**: Copy or insert code snippets from responses
- **Inline Diffs**: See proposed changes highlighted in your editor
- **Pending Changes View**: Review and apply/reject AI-suggested modifications

## Requirements

- VSCode 1.80.0 or later
- Animus server running locally (`animus ide`)

## Installation

### From VSIX (Development)

1. Build the extension:
   ```bash
   cd vscode-extension
   npm install
   npm run compile
   npx vsce package
   ```

2. Install the VSIX:
   - Open VSCode
   - Go to Extensions view (Ctrl+Shift+X)
   - Click "..." menu > "Install from VSIX..."
   - Select the generated `.vsix` file

### Start the Server

Before using the extension, start the Animus WebSocket server:

```bash
# Install websockets dependency
pip install websockets

# Start the IDE server
animus ide
```

The server runs on `ws://localhost:8765` by default.

## Configuration

Configure the extension in VSCode settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `animus.serverUrl` | `ws://localhost:8765` | WebSocket server URL |
| `animus.streamingEnabled` | `true` | Enable streaming responses |
| `animus.autoConnect` | `true` | Auto-connect on startup |

## Usage

### Starting a Chat

1. Click the Animus icon in the activity bar (left sidebar)
2. Type your message in the input box
3. Press Enter or click Send

### Context Features

The extension automatically includes context about:
- Current file path and language
- Selected text (if any)
- Workspace folder

### Working with Code

When the AI suggests code:
- **Copy**: Copy the code to clipboard
- **Insert**: Insert at cursor position

### Reviewing Changes

When the AI proposes file modifications:
1. Changes appear in the "Pending Changes" view
2. Click a change to see the diff
3. Use context menu to Accept or Reject

## Commands

| Command | Description |
|---------|-------------|
| `Animus: Start Chat` | Open the chat panel |
| `Animus: Connect to Server` | Connect to Animus server |
| `Animus: Disconnect from Server` | Disconnect from server |
| `Animus: Show Diff` | Show diff for a pending change |
| `Animus: Accept Change` | Apply a proposed change |
| `Animus: Reject Change` | Discard a proposed change |
| `Animus: Accept All Changes` | Apply all pending changes |
| `Animus: Reject All Changes` | Discard all pending changes |

## Keybindings

| Key | Command |
|-----|---------|
| `Ctrl+Shift+A` | Start Animus Chat |

## Architecture

```
VSCode Extension <--WebSocket--> Animus Server <--> LLM Provider
     |                               |
     v                               v
  Webview UI                    Tool Execution
  Tree Views                    File Operations
  Decorations                   Code Analysis
```

## Development

```bash
# Install dependencies
npm install

# Compile TypeScript
npm run compile

# Watch for changes
npm run watch

# Run linter
npm run lint
```

### Project Structure

```
vscode-extension/
├── src/
│   ├── extension.ts          # Extension entry point
│   ├── client.ts             # WebSocket client
│   ├── statusBar.ts          # Status bar manager
│   ├── views/
│   │   ├── chatView.ts       # Chat webview provider
│   │   └── changesView.ts    # Pending changes tree view
│   └── decorations/
│       └── diffDecorations.ts # Inline diff decorations
├── resources/
│   └── animus-icon.svg       # Activity bar icon
├── package.json              # Extension manifest
└── tsconfig.json             # TypeScript config
```

## Troubleshooting

### Connection Issues

1. Ensure the Animus server is running: `animus ide`
2. Check the server URL in settings
3. Look for errors in VSCode Developer Tools (Help > Toggle Developer Tools)

### No Response from AI

1. Verify an LLM model is configured in Animus
2. Check the server console for errors
3. Try disconnecting and reconnecting

## License

MIT
