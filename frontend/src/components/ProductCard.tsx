"use client";

import { useState } from "react";
import { SearchResult } from "@/utils/api";
import { Tag } from "lucide-react";

interface ProductCardProps {
  product: SearchResult;
}

export default function ProductCard({ product }: ProductCardProps) {
  const [imgError, setImgError] = useState(false);

  const similarityPct = Math.round(product.score * 100);
  const scoreColor =
    similarityPct >= 75 ? "text-green-600 bg-green-50" :
    similarityPct >= 50 ? "text-amber-600 bg-amber-50" :
    "text-gray-500 bg-gray-100";

  return (
    <div className="group bg-white border border-gray-100 rounded-2xl overflow-hidden hover:border-indigo-200 hover:shadow-md transition-all duration-200">
      {/* Image */}
      <div className="relative w-full h-52 bg-gray-50 overflow-hidden">
        {!imgError ? (
          <img
            src={product.image_url}
            alt={product.name}
            className="w-full h-full object-contain p-2 group-hover:scale-105 transition-transform duration-300"
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <Tag size={32} className="text-gray-300" />
          </div>
        )}
        {/* Similarity badge */}
        <span className={`absolute top-2 right-2 text-xs font-semibold px-2 py-0.5 rounded-full ${scoreColor}`}>
          {similarityPct}% match
        </span>
      </div>

      {/* Info */}
      <div className="p-3 space-y-1">
        <p className="text-xs text-indigo-500 font-medium uppercase tracking-wide truncate">
          {product.category}
        </p>
        <p className="text-sm font-medium text-gray-800 leading-snug line-clamp-2">
          {product.name}
        </p>
        {product.price != null && (
          <p className="text-sm font-semibold text-gray-900">
            ${product.price.toFixed(2)}
          </p>
        )}
        {product.description && product.description !== product.name && (
          <p className="text-xs text-gray-400 line-clamp-2">{product.description}</p>
        )}
      </div>
    </div>
  );
}
