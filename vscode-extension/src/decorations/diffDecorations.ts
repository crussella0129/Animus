/**
 * Inline diff decorations for showing proposed changes.
 */

import * as vscode from 'vscode';

interface DiffLine {
    type: 'added' | 'removed' | 'unchanged';
    lineNumber: number;
    content: string;
}

interface FileDiff {
    filePath: string;
    originalContent: string;
    newContent: string;
    diffLines: DiffLine[];
}

export class DiffDecorationProvider {
    private addedDecorationType: vscode.TextEditorDecorationType;
    private removedDecorationType: vscode.TextEditorDecorationType;
    private modifiedDecorationType: vscode.TextEditorDecorationType;
    private fileDiffs: Map<string, FileDiff> = new Map();
    private disposables: vscode.Disposable[] = [];

    constructor() {
        // Create decoration types
        this.addedDecorationType = vscode.window.createTextEditorDecorationType({
            backgroundColor: new vscode.ThemeColor('diffEditor.insertedTextBackground'),
            isWholeLine: true,
            overviewRulerColor: new vscode.ThemeColor('diffEditor.insertedLineBackground'),
            overviewRulerLane: vscode.OverviewRulerLane.Left,
            gutterIconPath: this.createGutterIcon('+', '#89d185'),
            gutterIconSize: 'contain'
        });

        this.removedDecorationType = vscode.window.createTextEditorDecorationType({
            backgroundColor: new vscode.ThemeColor('diffEditor.removedTextBackground'),
            isWholeLine: true,
            overviewRulerColor: new vscode.ThemeColor('diffEditor.removedLineBackground'),
            overviewRulerLane: vscode.OverviewRulerLane.Left,
            gutterIconPath: this.createGutterIcon('-', '#f85149'),
            gutterIconSize: 'contain'
        });

        this.modifiedDecorationType = vscode.window.createTextEditorDecorationType({
            backgroundColor: 'rgba(255, 200, 0, 0.1)',
            isWholeLine: true,
            overviewRulerColor: 'rgba(255, 200, 0, 0.5)',
            overviewRulerLane: vscode.OverviewRulerLane.Left
        });

        // Listen for editor changes
        this.disposables.push(
            vscode.window.onDidChangeActiveTextEditor((editor) => {
                if (editor) {
                    this.applyDecorations(editor);
                }
            })
        );

        this.disposables.push(
            vscode.workspace.onDidChangeTextDocument((event) => {
                // Clear decorations if the file was edited
                const filePath = event.document.uri.fsPath;
                if (this.fileDiffs.has(filePath)) {
                    this.clearDiff(filePath);
                }
            })
        );
    }

    private createGutterIcon(symbol: string, color: string): vscode.Uri {
        // Create a simple SVG icon for the gutter
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">
            <text x="8" y="12" text-anchor="middle" font-family="monospace" font-size="14" fill="${color}">${symbol}</text>
        </svg>`;

        return vscode.Uri.parse(`data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`);
    }

    showDiff(filePath: string, originalContent: string, newContent: string) {
        // Compute diff
        const diffLines = this.computeDiff(originalContent, newContent);

        this.fileDiffs.set(filePath, {
            filePath,
            originalContent,
            newContent,
            diffLines
        });

        // Apply decorations to active editor if it matches
        const editor = vscode.window.activeTextEditor;
        if (editor && editor.document.uri.fsPath === filePath) {
            this.applyDecorations(editor);
        }
    }

    clearDiff(filePath: string) {
        this.fileDiffs.delete(filePath);

        // Clear decorations from matching editor
        const editor = vscode.window.activeTextEditor;
        if (editor && editor.document.uri.fsPath === filePath) {
            editor.setDecorations(this.addedDecorationType, []);
            editor.setDecorations(this.removedDecorationType, []);
            editor.setDecorations(this.modifiedDecorationType, []);
        }
    }

    clearAll() {
        this.fileDiffs.clear();

        // Clear all decorations from all visible editors
        for (const editor of vscode.window.visibleTextEditors) {
            editor.setDecorations(this.addedDecorationType, []);
            editor.setDecorations(this.removedDecorationType, []);
            editor.setDecorations(this.modifiedDecorationType, []);
        }
    }

    private computeDiff(original: string, modified: string): DiffLine[] {
        const originalLines = original.split('\n');
        const modifiedLines = modified.split('\n');
        const result: DiffLine[] = [];

        // Simple line-by-line diff using longest common subsequence approach
        const lcs = this.longestCommonSubsequence(originalLines, modifiedLines);
        let origIdx = 0;
        let modIdx = 0;
        let lcsIdx = 0;

        while (origIdx < originalLines.length || modIdx < modifiedLines.length) {
            if (lcsIdx < lcs.length &&
                origIdx < originalLines.length &&
                modIdx < modifiedLines.length &&
                originalLines[origIdx] === lcs[lcsIdx] &&
                modifiedLines[modIdx] === lcs[lcsIdx]) {
                // Unchanged line
                result.push({
                    type: 'unchanged',
                    lineNumber: modIdx,
                    content: modifiedLines[modIdx]
                });
                origIdx++;
                modIdx++;
                lcsIdx++;
            } else if (origIdx < originalLines.length &&
                (lcsIdx >= lcs.length || originalLines[origIdx] !== lcs[lcsIdx])) {
                // Removed line
                result.push({
                    type: 'removed',
                    lineNumber: origIdx,
                    content: originalLines[origIdx]
                });
                origIdx++;
            } else if (modIdx < modifiedLines.length &&
                (lcsIdx >= lcs.length || modifiedLines[modIdx] !== lcs[lcsIdx])) {
                // Added line
                result.push({
                    type: 'added',
                    lineNumber: modIdx,
                    content: modifiedLines[modIdx]
                });
                modIdx++;
            }
        }

        return result;
    }

    private longestCommonSubsequence(a: string[], b: string[]): string[] {
        const m = a.length;
        const n = b.length;
        const dp: number[][] = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));

        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                if (a[i - 1] === b[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        // Backtrack to find the actual LCS
        const result: string[] = [];
        let i = m, j = n;
        while (i > 0 && j > 0) {
            if (a[i - 1] === b[j - 1]) {
                result.unshift(a[i - 1]);
                i--;
                j--;
            } else if (dp[i - 1][j] > dp[i][j - 1]) {
                i--;
            } else {
                j--;
            }
        }

        return result;
    }

    private applyDecorations(editor: vscode.TextEditor) {
        const filePath = editor.document.uri.fsPath;
        const fileDiff = this.fileDiffs.get(filePath);

        if (!fileDiff) {
            return;
        }

        const addedRanges: vscode.DecorationOptions[] = [];
        const removedRanges: vscode.DecorationOptions[] = [];
        const modifiedRanges: vscode.DecorationOptions[] = [];

        for (const line of fileDiff.diffLines) {
            if (line.lineNumber >= editor.document.lineCount) {
                continue;
            }

            const range = editor.document.lineAt(line.lineNumber).range;

            switch (line.type) {
                case 'added':
                    addedRanges.push({
                        range,
                        hoverMessage: new vscode.MarkdownString('**Added** by Animus')
                    });
                    break;
                case 'removed':
                    // Can't show removed lines in current content, show as modified
                    modifiedRanges.push({
                        range,
                        hoverMessage: new vscode.MarkdownString(`**Removed**:\n\`\`\`\n${line.content}\n\`\`\``)
                    });
                    break;
            }
        }

        editor.setDecorations(this.addedDecorationType, addedRanges);
        editor.setDecorations(this.removedDecorationType, removedRanges);
        editor.setDecorations(this.modifiedDecorationType, modifiedRanges);
    }

    dispose() {
        this.addedDecorationType.dispose();
        this.removedDecorationType.dispose();
        this.modifiedDecorationType.dispose();

        for (const disposable of this.disposables) {
            disposable.dispose();
        }
    }
}
