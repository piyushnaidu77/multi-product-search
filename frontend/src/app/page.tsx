"use client";

import SearchBar from "@/components/SearchBar";
import ResultsGrid from "@/components/ResultsGrid";
import { useSearch } from "@/hooks/useSearch";

export default function Home() {
  const {
    results, loading, error, queryType, total,
    runTextSearch, runImageSearch, runUrlSearch, reset,
  } = useSearch();

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-indigo-50">
      <div className="max-w-6xl mx-auto px-4 py-16 space-y-8">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="inline-flex items-center gap-2 bg-indigo-100 text-indigo-700 text-xs font-semibold px-3 py-1.5 rounded-full mb-2">
            <span>✦</span> Powered by CLIP + Pinecone
          </div>
          <h1 className="text-4xl font-bold text-gray-900 tracking-tight">
            Multimodal Product Search
          </h1>
          <p className="text-gray-500 text-base max-w-md mx-auto">
            Find products by describing them in words, uploading a photo, or pasting an image URL.
          </p>
        </div>

        {/* Search */}
        <SearchBar
          onTextSearch={runTextSearch}
          onImageSearch={runImageSearch}
          onUrlSearch={runUrlSearch}
          loading={loading}
          onReset={reset}
        />

        {/* Error */}
        {error && (
          <div className="max-w-2xl mx-auto bg-red-50 border border-red-200 text-red-700 rounded-xl px-4 py-3 text-sm">
            {error}
          </div>
        )}

        {/* Results */}
        <ResultsGrid results={results} total={total} queryType={queryType} loading={loading} />
      </div>
    </main>
  );
}
