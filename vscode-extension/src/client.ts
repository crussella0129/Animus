/**
 * WebSocket client for communicating with Animus server.
 */

import * as vscode from 'vscode';
import WebSocket from 'ws';
import { EventEmitter } from 'events';

export interface ClientConfig {
    serverUrl: string;
    streamingEnabled: boolean;
}

export interface AnimusMessage {
    type: string;
    content?: string;
    data?: any;
    error?: string;
}

export class AnimusClient extends EventEmitter {
    private ws: WebSocket | null = null;
    private config: ClientConfig;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectDelay = 1000;
    private messageQueue: AnimusMessage[] = [];
    private isConnecting = false;

    constructor() {
        super();
        const vsConfig = vscode.workspace.getConfiguration('animus');
        this.config = {
            serverUrl: vsConfig.get('serverUrl', 'ws://localhost:8765'),
            streamingEnabled: vsConfig.get('streamingEnabled', true)
        };
    }

    updateConfig(config: Partial<ClientConfig>) {
        this.config = { ...this.config, ...config };
    }

    async connect(): Promise<void> {
        if (this.ws?.readyState === WebSocket.OPEN) {
            return;
        }

        if (this.isConnecting) {
            return;
        }

        this.isConnecting = true;

        return new Promise((resolve, reject) => {
            try {
                this.ws = new WebSocket(this.config.serverUrl);

                this.ws.on('open', () => {
                    console.log('Connected to Animus server');
                    this.isConnecting = false;
                    this.reconnectAttempts = 0;
                    this.emit('connected');

                    // Send queued messages
                    while (this.messageQueue.length > 0) {
                        const msg = this.messageQueue.shift();
                        if (msg) {
                            this.send(msg);
                        }
                    }

                    resolve();
                });

                this.ws.on('message', (data: WebSocket.Data) => {
                    try {
                        const message = JSON.parse(data.toString());
                        this.handleMessage(message);
                    } catch (error) {
                        console.error('Failed to parse message:', error);
                    }
                });

                this.ws.on('close', () => {
                    console.log('Disconnected from Animus server');
                    this.isConnecting = false;
                    this.emit('disconnected');
                    this.attemptReconnect();
                });

                this.ws.on('error', (error) => {
                    console.error('WebSocket error:', error);
                    this.isConnecting = false;
                    this.emit('error', error);
                    reject(error);
                });

            } catch (error) {
                this.isConnecting = false;
                reject(error);
            }
        });
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.reconnectAttempts = this.maxReconnectAttempts; // Prevent auto-reconnect
    }

    private attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

        console.log(`Attempting reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);

        setTimeout(() => {
            this.connect().catch(() => {
                // Reconnect failed, will try again
            });
        }, delay);
    }

    private handleMessage(message: AnimusMessage) {
        switch (message.type) {
            case 'token':
                // Streaming token
                this.emit('token', message.content);
                break;

            case 'response':
                // Complete response
                this.emit('message', message);
                break;

            case 'file_change':
                // File modification proposed
                this.emit('fileChange', message.data);
                break;

            case 'tool_call':
                // Tool being executed
                this.emit('toolCall', message.data);
                break;

            case 'tool_result':
                // Tool execution result
                this.emit('toolResult', message.data);
                break;

            case 'error':
                this.emit('error', new Error(message.error || 'Unknown error'));
                break;

            case 'status':
                this.emit('status', message.data);
                break;

            default:
                console.log('Unknown message type:', message.type);
                this.emit('message', message);
        }
    }

    send(message: AnimusMessage) {
        if (this.ws?.readyState !== WebSocket.OPEN) {
            // Queue message for when connection is established
            this.messageQueue.push(message);
            this.connect().catch(() => {});
            return;
        }

        this.ws.send(JSON.stringify(message));
    }

    async sendChat(content: string, context?: ChatContext): Promise<void> {
        this.send({
            type: 'chat',
            content,
            data: {
                streaming: this.config.streamingEnabled,
                context
            }
        });
    }

    async sendCommand(command: string, args?: any): Promise<void> {
        this.send({
            type: 'command',
            content: command,
            data: args
        });
    }

    isConnected(): boolean {
        return this.ws?.readyState === WebSocket.OPEN;
    }
}

export interface ChatContext {
    filePath?: string;
    selection?: {
        text: string;
        startLine: number;
        endLine: number;
    };
    language?: string;
    workspaceFolder?: string;
}
