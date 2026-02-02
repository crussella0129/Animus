/**
 * Chat webview provider for Animus conversations.
 */

import * as vscode from 'vscode';
import { AnimusClient, ChatContext } from '../client';

export class ChatViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'animus.chatView';

    private view?: vscode.WebviewView;
    private client: AnimusClient;
    private messages: ChatMessage[] = [];
    private isStreaming = false;
    private currentStreamContent = '';

    constructor(
        private readonly extensionUri: vscode.Uri,
        client: AnimusClient
    ) {
        this.client = client;
    }

    resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ) {
        this.view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this.extensionUri]
        };

        webviewView.webview.html = this.getHtmlContent(webviewView.webview);

        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'sendMessage':
                    await this.sendMessage(data.message);
                    break;
                case 'clearChat':
                    this.clearChat();
                    break;
                case 'copyCode':
                    await vscode.env.clipboard.writeText(data.code);
                    vscode.window.showInformationMessage('Code copied to clipboard');
                    break;
                case 'insertCode':
                    await this.insertCodeAtCursor(data.code);
                    break;
            }
        });

        // Restore previous messages
        this.updateWebview();
    }

    async sendMessage(content: string) {
        if (!content.trim()) return;

        // Add user message
        this.addMessage({
            role: 'user',
            content,
            timestamp: new Date()
        });

        // Get context from active editor
        const context = this.getEditorContext();

        // Send to server
        this.isStreaming = true;
        this.currentStreamContent = '';

        // Add placeholder for assistant response
        this.addMessage({
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            isStreaming: true
        });

        try {
            await this.client.sendChat(content, context);
        } catch (error) {
            this.handleError(error);
        }
    }

    handleServerMessage(message: any) {
        if (message.type === 'response') {
            // Complete response received
            this.isStreaming = false;

            // Update the last assistant message
            const lastMessage = this.messages[this.messages.length - 1];
            if (lastMessage?.role === 'assistant') {
                lastMessage.content = message.content || this.currentStreamContent;
                lastMessage.isStreaming = false;
            }

            this.currentStreamContent = '';
            this.updateWebview();
        }
    }

    handleStreamingToken(token: string) {
        this.currentStreamContent += token;

        // Update the last assistant message with streaming content
        const lastMessage = this.messages[this.messages.length - 1];
        if (lastMessage?.role === 'assistant' && lastMessage.isStreaming) {
            lastMessage.content = this.currentStreamContent;
            this.updateWebview();
        }
    }

    private handleError(error: any) {
        this.isStreaming = false;

        // Update last message to show error
        const lastMessage = this.messages[this.messages.length - 1];
        if (lastMessage?.role === 'assistant' && lastMessage.isStreaming) {
            lastMessage.content = `Error: ${error.message || error}`;
            lastMessage.isStreaming = false;
            lastMessage.isError = true;
        }

        this.updateWebview();
    }

    private addMessage(message: ChatMessage) {
        this.messages.push(message);
        this.updateWebview();
    }

    private clearChat() {
        this.messages = [];
        this.updateWebview();
    }

    private getEditorContext(): ChatContext | undefined {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return undefined;

        const selection = editor.selection;
        const document = editor.document;

        return {
            filePath: document.uri.fsPath,
            language: document.languageId,
            workspaceFolder: vscode.workspace.getWorkspaceFolder(document.uri)?.uri.fsPath,
            selection: !selection.isEmpty ? {
                text: document.getText(selection),
                startLine: selection.start.line,
                endLine: selection.end.line
            } : undefined
        };
    }

    private async insertCodeAtCursor(code: string) {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }

        await editor.edit((editBuilder) => {
            editBuilder.insert(editor.selection.active, code);
        });
    }

    private updateWebview() {
        if (this.view) {
            this.view.webview.postMessage({
                type: 'updateMessages',
                messages: this.messages
            });
        }
    }

    private getHtmlContent(webview: vscode.Webview): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'unsafe-inline';">
    <title>Animus Chat</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            display: flex;
            flex-direction: column;
            height: 100vh;
            padding: 8px;
        }

        #chat-container {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 8px;
        }

        .message {
            padding: 8px 12px;
            margin-bottom: 8px;
            border-radius: 8px;
            max-width: 90%;
        }

        .message.user {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            margin-left: auto;
        }

        .message.assistant {
            background-color: var(--vscode-editor-inactiveSelectionBackground);
        }

        .message.error {
            background-color: var(--vscode-inputValidation-errorBackground);
            border: 1px solid var(--vscode-inputValidation-errorBorder);
        }

        .message-content {
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .message-content code {
            font-family: var(--vscode-editor-font-family);
            background-color: var(--vscode-textCodeBlock-background);
            padding: 2px 4px;
            border-radius: 3px;
        }

        .message-content pre {
            background-color: var(--vscode-textCodeBlock-background);
            padding: 8px;
            border-radius: 4px;
            overflow-x: auto;
            margin: 8px 0;
            position: relative;
        }

        .code-actions {
            position: absolute;
            top: 4px;
            right: 4px;
            display: flex;
            gap: 4px;
        }

        .code-action-btn {
            background: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
            border: none;
            padding: 2px 6px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        }

        .code-action-btn:hover {
            background: var(--vscode-button-secondaryHoverBackground);
        }

        .streaming-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: var(--vscode-progressBar-background);
            border-radius: 50%;
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        #input-container {
            display: flex;
            gap: 8px;
        }

        #message-input {
            flex: 1;
            padding: 8px;
            border: 1px solid var(--vscode-input-border);
            background-color: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border-radius: 4px;
            font-family: inherit;
            font-size: inherit;
            resize: none;
            min-height: 36px;
            max-height: 150px;
        }

        #message-input:focus {
            outline: 1px solid var(--vscode-focusBorder);
        }

        #send-btn {
            padding: 8px 16px;
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }

        #send-btn:hover {
            background-color: var(--vscode-button-hoverBackground);
        }

        #send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--vscode-panel-border);
        }

        .header-title {
            font-weight: bold;
        }

        .clear-btn {
            background: none;
            border: none;
            color: var(--vscode-foreground);
            cursor: pointer;
            opacity: 0.7;
        }

        .clear-btn:hover {
            opacity: 1;
        }

        .empty-state {
            text-align: center;
            color: var(--vscode-descriptionForeground);
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <span class="header-title">Animus</span>
        <button class="clear-btn" onclick="clearChat()" title="Clear chat">Clear</button>
    </div>

    <div id="chat-container">
        <div class="empty-state" id="empty-state">
            Start a conversation with Animus
        </div>
    </div>

    <div id="input-container">
        <textarea
            id="message-input"
            placeholder="Ask Animus anything..."
            rows="1"
        ></textarea>
        <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        let messages = [];

        // Handle messages from extension
        window.addEventListener('message', event => {
            const message = event.data;
            switch (message.type) {
                case 'updateMessages':
                    messages = message.messages;
                    renderMessages();
                    break;
            }
        });

        function renderMessages() {
            const container = document.getElementById('chat-container');
            const emptyState = document.getElementById('empty-state');

            if (messages.length === 0) {
                emptyState.style.display = 'block';
                container.innerHTML = '';
                container.appendChild(emptyState);
                return;
            }

            emptyState.style.display = 'none';
            container.innerHTML = messages.map((msg, index) => {
                const classes = ['message', msg.role];
                if (msg.isError) classes.push('error');

                let content = escapeHtml(msg.content);
                content = formatMarkdown(content);

                const streamingIndicator = msg.isStreaming
                    ? '<span class="streaming-indicator"></span>'
                    : '';

                return \`<div class="\${classes.join(' ')}">
                    <div class="message-content">\${content}\${streamingIndicator}</div>
                </div>\`;
            }).join('');

            // Scroll to bottom
            container.scrollTop = container.scrollHeight;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function formatMarkdown(text) {
            // Simple markdown formatting
            // Code blocks
            text = text.replace(/\`\`\`(\\w*)\\n([\\s\\S]*?)\`\`\`/g, (match, lang, code) => {
                const codeId = 'code-' + Math.random().toString(36).substr(2, 9);
                return \`<pre data-code-id="\${codeId}"><div class="code-actions">
                    <button class="code-action-btn" onclick="copyCode('\${codeId}')">Copy</button>
                    <button class="code-action-btn" onclick="insertCode('\${codeId}')">Insert</button>
                </div><code id="\${codeId}">\${code.trim()}</code></pre>\`;
            });

            // Inline code
            text = text.replace(/\`([^\`]+)\`/g, '<code>$1</code>');

            // Bold
            text = text.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');

            // Italic
            text = text.replace(/\\*([^*]+)\\*/g, '<em>$1</em>');

            return text;
        }

        function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();

            if (message) {
                vscode.postMessage({ type: 'sendMessage', message });
                input.value = '';
                input.style.height = 'auto';
            }
        }

        function clearChat() {
            vscode.postMessage({ type: 'clearChat' });
        }

        function copyCode(codeId) {
            const codeElement = document.getElementById(codeId);
            if (codeElement) {
                vscode.postMessage({ type: 'copyCode', code: codeElement.textContent });
            }
        }

        function insertCode(codeId) {
            const codeElement = document.getElementById(codeId);
            if (codeElement) {
                vscode.postMessage({ type: 'insertCode', code: codeElement.textContent });
            }
        }

        // Handle Enter to send
        document.getElementById('message-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Auto-resize textarea
        document.getElementById('message-input').addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        });
    </script>
</body>
</html>`;
    }
}

interface ChatMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
    isStreaming?: boolean;
    isError?: boolean;
}
