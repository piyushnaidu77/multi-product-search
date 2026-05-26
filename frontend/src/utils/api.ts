import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

const api = axios.create({ baseURL: API_BASE });

export interface SearchResult {
  id: string;
  score: number;
  name: string;
  category: string;
  price: number | null;
  image_url: string;
  description: string;
}

export interface SearchResponse {
  results: SearchResult[];
  query_type: string;
  total: number;
}

/**
 * Search by text query (natural language)
 */
export async function searchByText(
  query: string,
  top_k = 12,
  category?: string
): Promise<SearchResponse> {
  const { data } = await api.post<SearchResponse>("/search/text", {
    query,
    top_k,
    category: category || undefined,
  });
  return data;
}

/**
 * Search by uploading an image file
 */
export async function searchByImageFile(
  file: File,
  top_k = 12,
  category?: string
): Promise<SearchResponse> {
  const form = new FormData();
  form.append("file", file);
  const params = new URLSearchParams({ top_k: String(top_k) });
  if (category) params.set("category", category);

  const { data } = await api.post<SearchResponse>(
    `/search/image?${params}`,
    form,
    { headers: { "Content-Type": "multipart/form-data" } }
  );
  return data;
}

/**
 * Search by a public image URL
 */
export async function searchByImageUrl(
  image_url: string,
  top_k = 12,
  category?: string
): Promise<SearchResponse> {
  const { data } = await api.post<SearchResponse>("/search/url", {
    image_url,
    top_k,
    category: category || undefined,
  });
  return data;
}
