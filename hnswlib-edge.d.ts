export type SpaceType = 'IP' | 'L2' | 'GEODEGREES';
export type VectorType = 'FLOAT32' | 'FLOAT16' | 'BFLOAT16';

export type MetadataValue = number | string | number[] | string[];
export type Metadata = Record<string, MetadataValue>;

export interface IndexSettings {
  dimension: number;
  spaceType?: SpaceType;
  vectorType?: VectorType;
  M?: number;
  efConstruction?: number;
  mrlScanDim?: number;
  /**
   * Numeric fields that are always stored as floats. JavaScript can't tell 3 from 3.0,
   * so list any float field that may hold whole numbers to keep range filters correct.
   */
  doubleFields?: string[];
  [key: string]: unknown;
}

export interface IndexStatus {
  currentElements: number;
  maxElements: number;
  deletedElements: number;
}

export interface AddDocumentsRequest {
  ids: ArrayLike<number>;
  /** One vector per id, or a flat Float32Array of ids.length * dimension values. */
  vectors: ArrayLike<number>[] | Float32Array;
  metadatas?: (Metadata | null)[];
}

export interface SearchOptions {
  k?: number;
  /** Skip this many nearest hits before returning k, for pagination. */
  offset?: number;
  efSearch?: number;
  /** Filter DSL string, same syntax as the server. */
  filter?: string;
  returnMetadata?: boolean;
  /** MRL indexes only: rerank this many scan-dim candidates at full dimensionality. */
  rerankSize?: number;
}

export interface SimilarOptions extends SearchOptions {
  /** Leave the input document out of the results. Defaults to true. */
  excludeInputDocument?: boolean;
}

export interface SearchResult {
  hits: number[];
  distances: number[];
  metadatas?: Metadata[];
}

export interface Document {
  id: number;
  vector: Float32Array;
  metadata: Metadata;
}

export interface SavedIndex {
  bin: Uint8Array;
  settings: IndexSettings;
  data: Uint8Array;
}

export interface LoadSource {
  bin: Uint8Array | ArrayBuffer | ArrayBufferView;
  settings: IndexSettings | string;
  data?: Uint8Array | ArrayBuffer | ArrayBufferView | null;
}

export interface FromUrlOptions {
  fetch?: typeof fetch;
  requestInit?: RequestInit;
}

/** Release version (matches the server's /version). */
export const version: string;

/** Emscripten module options, e.g. { locateFile: (file) => `/assets/${file}` }. */
export function init(options?: Record<string, unknown>): Promise<unknown>;

export class VectorIndex {
  private constructor();
  static create(settings: IndexSettings | string, options?: { initialCapacity?: number }): Promise<VectorIndex>;
  static load(source: LoadSource): Promise<VectorIndex>;
  static fromUrl(baseUrl: string | URL, name: string, options?: FromUrlOptions): Promise<VectorIndex>;

  readonly dimension: number;
  readonly settings: IndexSettings;
  status(): IndexStatus;
  addDocuments(request: AddDocumentsRequest): void;
  deleteDocuments(ids: ArrayLike<number>): void;
  search(queryVector: ArrayLike<number>, options?: SearchOptions): SearchResult;
  similar(id: number, options?: SimilarOptions): SearchResult;
  contains(id: number): boolean;
  getDocument(id: number): Document | null;
  save(): SavedIndex;
  dispose(): void;
  [Symbol.dispose](): void;
}
