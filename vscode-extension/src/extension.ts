/**
 * Animus VSCode Extension
 *
 * Provides IDE integration for the Animus AI coding assistant.
 */

import * as vscode from 'vscode';
import { AnimusClient } from './client';
import { ChatViewProvider } from './views/chatView';
import { ChangesViewProvider } from './views/changesView';
import { DiffDecorationProvider } from './decorations/diffDecorations';
import { StatusBarManager } from './statusBar';

let client: AnimusClient;
let chatViewProvider: ChatViewProvider;
let changesViewProvider: ChangesViewProvider;
let diffDecorations: DiffDecorationProvider;
let statusBar: StatusBarManager;

export async function activate(context: vscode.ExtensionContext) {
    console.log('Animus extension activating...');

    // Initialize components
    client = new AnimusClient();
    chatViewProvider = new ChatViewProvider(context.extensionUri, client);
    changesViewProvider = new ChangesViewProvider();
    diffDecorations = new DiffDecorationProvider();
    statusBar = new StatusBarManager();

    // Register chat webview provider
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            'animus.chatView',
            chatViewProvider
        )
    );

    // Register changes tree view provider
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider(
            'animus.changesView',
            changesViewProvider
        )
    );

    // Register commands
    registerCommands(context);

    // Set up event listeners
    setupEventListeners(context);

    // Auto-connect if enabled
    const config = vscode.workspace.getConfiguration('animus');
    if (config.get('autoConnect', true)) {
        try {
            await client.connect();
            statusBar.setConnected(true);
            vscode.commands.executeCommand('setContext', 'animus.connected', true);
        } catch (error) {
            console.log('Auto-connect failed, will retry when user initiates');
        }
    }

    console.log('Animus extension activated');
}

function registerCommands(context: vscode.ExtensionContext) {
    // Chat commands
    context.subscriptions.push(
        vscode.commands.registerCommand('animus.startChat', () => {
            vscode.commands.executeCommand('animus.chatView.focus');
        })
    );

    // Selection-based commands
    context.subscriptions.push(
        vscode.commands.registerCommand('animus.askAboutSelection', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.document.getText(editor.selection);
            if (!selection) {
                vscode.window.showWarningMessage('Please select some code first');
                return;
            }

            const question = await vscode.window.showInputBox({
                prompt: 'What would you like to know about this code?',
                placeHolder: 'Enter your question...'
            });

            if (question) {
                await chatViewProvider.sendMessage(
                    `About this code:\n\`\`\`\n${selection}\n\`\`\`\n\n${question}`
                );
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('animus.explainCode', async () => {
            await executeCodeCommand('Explain this code in detail:');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('animus.fixCode', async () => {
            await executeCodeCommand('Fix any issues in this code:');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('animus.refactorCode', async () => {
            await executeCodeCommand('Refactor this code to improve it:');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('animus.generateTests', async () => {
            await executeCodeCommand('Generate unit tests for this code:');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('animus.addDocumentation', async () => {
            await executeCodeCommand('Add documentation/comments to this code:');
        })
    );

    // Connection commands
    context.subscriptions.push(
        vscode.commands.registerCommand('animus.connectServer', async () => {
            try {
                await client.connect();
                statusBar.setConnected(true);
                vscode.commands.executeCommand('setContext', 'animus.connected', true);
                vscode.window.showInformationMessage('Connected to Animus server');
            } catch (error) {
                vscode.window.showErrorMessage(`Failed to connect: ${error}`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('animus.disconnectServer', () => {
            client.disconnect();
            statusBar.setConnected(false);
            vscode.commands.executeCommand('setContext', 'animus.connected', false);
            vscode.window.showInformationMessage('Disconnected from Animus server');
        })
    );

    // Diff commands
    context.subscriptions.push(
        vscode.commands.registerCommand('animus.showDiff', async () => {
            const changes = changesViewProvider.getChanges();
            if (changes.length === 0) {
                vscode.window.showInformationMessage('No pending changes');
                return;
            }
            // Show diff for first change
            await showDiffForChange(changes[0]);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('animus.acceptChanges', async () => {
            const changes = changesViewProvider.getChanges();
            if (changes.length === 0) {
                vscode.window.showInformationMessage('No pending changes');
                return;
            }

            const config = vscode.workspace.getConfiguration('animus');
            if (config.get('confirmBeforeApply', true)) {
                const confirm = await vscode.window.showWarningMessage(
                    `Apply ${changes.length} pending change(s)?`,
                    'Apply All',
                    'Cancel'
                );
                if (confirm !== 'Apply All') return;
            }

            await applyAllChanges(changes);
            changesViewProvider.clearChanges();
            diffDecorations.clearAll();
            vscode.window.showInformationMessage('Changes applied');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('animus.rejectChanges', () => {
            changesViewProvider.clearChanges();
            diffDecorations.clearAll();
            vscode.window.showInformationMessage('Changes rejected');
        })
    );
}

async function executeCodeCommand(prompt: string) {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
    }

    const selection = editor.document.getText(editor.selection);
    if (!selection) {
        vscode.window.showWarningMessage('Please select some code first');
        return;
    }

    const language = editor.document.languageId;
    const filePath = editor.document.uri.fsPath;

    await chatViewProvider.sendMessage(
        `${prompt}\n\nFile: ${filePath}\nLanguage: ${language}\n\n\`\`\`${language}\n${selection}\n\`\`\``
    );
}

async function showDiffForChange(change: any) {
    const originalUri = vscode.Uri.parse(`animus-original:${change.filePath}`);
    const modifiedUri = vscode.Uri.file(change.filePath);

    await vscode.commands.executeCommand(
        'vscode.diff',
        originalUri,
        modifiedUri,
        `Animus: ${change.filePath}`
    );
}

async function applyAllChanges(changes: any[]) {
    for (const change of changes) {
        try {
            const uri = vscode.Uri.file(change.filePath);
            const edit = new vscode.WorkspaceEdit();

            // Replace entire file content
            const document = await vscode.workspace.openTextDocument(uri);
            const fullRange = new vscode.Range(
                document.positionAt(0),
                document.positionAt(document.getText().length)
            );
            edit.replace(uri, fullRange, change.newContent);

            await vscode.workspace.applyEdit(edit);
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to apply change to ${change.filePath}: ${error}`);
        }
    }
}

function setupEventListeners(context: vscode.ExtensionContext) {
    // Listen for client events
    client.on('connected', () => {
        statusBar.setConnected(true);
        vscode.commands.executeCommand('setContext', 'animus.connected', true);
    });

    client.on('disconnected', () => {
        statusBar.setConnected(false);
        vscode.commands.executeCommand('setContext', 'animus.connected', false);
    });

    client.on('message', (message: any) => {
        chatViewProvider.handleServerMessage(message);
    });

    client.on('token', (token: string) => {
        chatViewProvider.handleStreamingToken(token);
    });

    client.on('fileChange', (change: any) => {
        changesViewProvider.addChange(change);

        const config = vscode.workspace.getConfiguration('animus');
        if (config.get('showInlineDiffs', true)) {
            diffDecorations.showDiff(change.filePath, change.originalContent, change.newContent);
        }
    });

    client.on('error', (error: Error) => {
        vscode.window.showErrorMessage(`Animus error: ${error.message}`);
    });

    // Listen for configuration changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration((e) => {
            if (e.affectsConfiguration('animus')) {
                // Reload configuration
                const config = vscode.workspace.getConfiguration('animus');
                client.updateConfig({
                    serverUrl: config.get('serverUrl', 'ws://localhost:8765'),
                    streamingEnabled: config.get('streamingEnabled', true)
                });
            }
        })
    );
}

export function deactivate() {
    if (client) {
        client.disconnect();
    }
    if (diffDecorations) {
        diffDecorations.dispose();
    }
    if (statusBar) {
        statusBar.dispose();
    }
}
