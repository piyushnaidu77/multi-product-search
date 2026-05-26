"use client";

import { SearchResult } from "@/utils/api";
import ProductCard from "./ProductCard";
import { SearchX } from "lucide-react";

interface ResultsGridProps {
  results: SearchResult[];
  total: number;
  queryType: string | null;
  loading: boolean;
}

export default function ResultsGrid({ results, total, queryType, loading }: ResultsGridProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mt-8">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="bg-gray-100 rounded-2xl h-72 animate-pulse" />
        ))}
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-gray-400 gap-3">
        <SearchX size={40} />
        <p className="text-sm">No results yet. Try a text or image search.</p>
      </div>
    );
  }

  const modeLabel =
    queryType === "text" ? "text query" :
    queryType === "image" ? "uploaded image" :
    "image URL";

  return (
    <div className="mt-8 space-y-4">
      <p className="text-sm text-gray-500">
        Found <span className="font-semibold text-gray-800">{total} products</span> matching your {modeLabel}
      </p>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {results.map((product) => (
          <ProductCard key={product.id} product={product} />
        ))}
      </div>
    </div>
  );
}
