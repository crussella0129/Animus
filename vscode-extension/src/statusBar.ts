/**
 * Status bar manager for showing Animus connection status.
 */

import * as vscode from 'vscode';

export class StatusBarManager {
    private statusBarItem: vscode.StatusBarItem;
    private isConnected = false;

    constructor() {
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            100
        );

        this.statusBarItem.command = 'animus.startChat';
        this.updateStatusBar();
        this.statusBarItem.show();
    }

    setConnected(connected: boolean) {
        this.isConnected = connected;
        this.updateStatusBar();
    }

    private updateStatusBar() {
        if (this.isConnected) {
            this.statusBarItem.text = '$(comment-discussion) Animus';
            this.statusBarItem.tooltip = 'Animus - Connected (click to chat)';
            this.statusBarItem.backgroundColor = undefined;
        } else {
            this.statusBarItem.text = '$(circle-slash) Animus';
            this.statusBarItem.tooltip = 'Animus - Disconnected (click to connect)';
            this.statusBarItem.backgroundColor = new vscode.ThemeColor(
                'statusBarItem.warningBackground'
            );
            this.statusBarItem.command = 'animus.connectServer';
        }
    }

    dispose() {
        this.statusBarItem.dispose();
    }
}
