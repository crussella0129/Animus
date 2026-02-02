/**
 * Tree view provider for pending changes from Animus.
 */

import * as vscode from 'vscode';
import * as path from 'path';

export interface PendingChange {
    id: string;
    filePath: string;
    originalContent: string;
    newContent: string;
    description: string;
    timestamp: Date;
}

export class ChangesViewProvider implements vscode.TreeDataProvider<ChangeItem> {
    private changes: PendingChange[] = [];
    private _onDidChangeTreeData = new vscode.EventEmitter<ChangeItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    getTreeItem(element: ChangeItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: ChangeItem): Thenable<ChangeItem[]> {
        if (element) {
            return Promise.resolve([]);
        }

        if (this.changes.length === 0) {
            return Promise.resolve([
                new ChangeItem(
                    'No pending changes',
                    '',
                    vscode.TreeItemCollapsibleState.None,
                    true
                )
            ]);
        }

        return Promise.resolve(
            this.changes.map(change => new ChangeItem(
                path.basename(change.filePath),
                change.description,
                vscode.TreeItemCollapsibleState.None,
                false,
                change
            ))
        );
    }

    addChange(change: PendingChange) {
        // Check if change for this file already exists
        const existingIndex = this.changes.findIndex(c => c.filePath === change.filePath);
        if (existingIndex >= 0) {
            this.changes[existingIndex] = change;
        } else {
            this.changes.push(change);
        }
        this._onDidChangeTreeData.fire(undefined);
    }

    removeChange(filePath: string) {
        this.changes = this.changes.filter(c => c.filePath !== filePath);
        this._onDidChangeTreeData.fire(undefined);
    }

    clearChanges() {
        this.changes = [];
        this._onDidChangeTreeData.fire(undefined);
    }

    getChanges(): PendingChange[] {
        return [...this.changes];
    }

    getChange(filePath: string): PendingChange | undefined {
        return this.changes.find(c => c.filePath === filePath);
    }
}

class ChangeItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly description: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly isEmpty: boolean,
        public readonly change?: PendingChange
    ) {
        super(label, collapsibleState);

        if (isEmpty) {
            this.iconPath = new vscode.ThemeIcon('info');
        } else if (change) {
            this.iconPath = new vscode.ThemeIcon('diff');
            this.tooltip = `${change.filePath}\n${change.description}`;
            this.command = {
                command: 'animus.showDiff',
                title: 'Show Diff',
                arguments: [change]
            };
            this.contextValue = 'pendingChange';
        }
    }
}
