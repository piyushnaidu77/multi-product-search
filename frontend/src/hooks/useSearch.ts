import { useState, useCallback } from "react";
import {
  searchByText,
  searchByImageFile,
  searchByImageUrl,
  SearchResult,
} from "@/utils/api";

type SearchMode = "text" | "image";

interface UseSearchReturn {
  results: SearchResult[];
  loading: boolean;
  error: string | null;
  queryType: string | null;
  total: number;
  runTextSearch: (query: string, category?: string) => Promise<void>;
  runImageSearch: (file: File, category?: string) => Promise<void>;
  runUrlSearch: (url: string, category?: string) => Promise<void>;
  reset: () => void;
}

export function useSearch(): UseSearchReturn {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [queryType, setQueryType] = useState<string | null>(null);
  const [total, setTotal] = useState(0);

  const handle = useCallback(async (fn: () => Promise<any>) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fn();
      setResults(data.results);
      setQueryType(data.query_type);
      setTotal(data.total);
    } catch (e: any) {
      setError(e?.response?.data?.detail || "Something went wrong. Is the API running?");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const runTextSearch = useCallback(
    (query: string, category?: string) =>
      handle(() => searchByText(query, 12, category)),
    [handle]
  );

  const runImageSearch = useCallback(
    (file: File, category?: string) =>
      handle(() => searchByImageFile(file, 12, category)),
    [handle]
  );

  const runUrlSearch = useCallback(
    (url: string, category?: string) =>
      handle(() => searchByImageUrl(url, 12, category)),
    [handle]
  );

  const reset = useCallback(() => {
    setResults([]);
    setError(null);
    setQueryType(null);
    setTotal(0);
  }, []);

  return { results, loading, error, queryType, total, runTextSearch, runImageSearch, runUrlSearch, reset };
}
