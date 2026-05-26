"use client";

import { useState, useRef } from "react";
import { useDropzone } from "react-dropzone";
import { Search, Upload, Link, X, Loader2 } from "lucide-react";
import clsx from "clsx";

type Mode = "text" | "image" | "url";

interface SearchBarProps {
  onTextSearch: (query: string, category?: string) => void;
  onImageSearch: (file: File, category?: string) => void;
  onUrlSearch: (url: string, category?: string) => void;
  loading: boolean;
  onReset: () => void;
}

const CATEGORIES = ["All", "Clothing", "Shoes", "Electronics", "Furniture", "Beauty", "Sports"];

export default function SearchBar({
  onTextSearch,
  onImageSearch,
  onUrlSearch,
  loading,
  onReset,
}: SearchBarProps) {
  const [mode, setMode] = useState<Mode>("text");
  const [textQuery, setTextQuery] = useState("");
  const [urlQuery, setUrlQuery] = useState("");
  const [category, setCategory] = useState("All");
  const [previewFile, setPreviewFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "image/*": [".jpeg", ".jpg", ".png", ".webp"] },
    maxFiles: 1,
    onDrop: (files) => {
      if (files[0]) {
        setPreviewFile(files[0]);
        setPreviewUrl(URL.createObjectURL(files[0]));
      }
    },
  });

  const resolvedCategory = category === "All" ? undefined : category;

  const handleSubmit = () => {
    if (loading) return;
    if (mode === "text" && textQuery.trim()) onTextSearch(textQuery.trim(), resolvedCategory);
    if (mode === "image" && previewFile) onImageSearch(previewFile, resolvedCategory);
    if (mode === "url" && urlQuery.trim()) onUrlSearch(urlQuery.trim(), resolvedCategory);
  };

  const handleReset = () => {
    setTextQuery("");
    setUrlQuery("");
    setPreviewFile(null);
    setPreviewUrl(null);
    onReset();
  };

  const canSubmit =
    !loading &&
    ((mode === "text" && textQuery.trim().length > 0) ||
      (mode === "image" && previewFile !== null) ||
      (mode === "url" && urlQuery.trim().length > 0));

  return (
    <div className="w-full max-w-2xl mx-auto space-y-4">
      {/* Mode toggle */}
      <div className="flex gap-1 bg-gray-100 rounded-xl p-1 w-fit mx-auto">
        {(["text", "image", "url"] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={clsx(
              "flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all",
              mode === m
                ? "bg-white text-indigo-600 shadow-sm"
                : "text-gray-500 hover:text-gray-800"
            )}
          >
            {m === "text" && <Search size={14} />}
            {m === "image" && <Upload size={14} />}
            {m === "url" && <Link size={14} />}
            {m === "text" ? "Text" : m === "image" ? "Upload" : "Image URL"}
          </button>
        ))}
      </div>

      {/* Input area */}
      <div className="space-y-3">
        {mode === "text" && (
          <div className="flex gap-2">
            <input
              type="text"
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              placeholder="e.g. red running shoes for women..."
              className="flex-1 border border-gray-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
            />
          </div>
        )}

        {mode === "url" && (
          <div className="flex gap-2">
            <input
              type="url"
              value={urlQuery}
              onChange={(e) => setUrlQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              placeholder="https://example.com/product-image.jpg"
              className="flex-1 border border-gray-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
            />
          </div>
        )}

        {mode === "image" && (
          <div>
            {previewUrl ? (
              <div className="relative w-full h-40 rounded-xl overflow-hidden border border-gray-200">
                <img src={previewUrl} alt="Preview" className="w-full h-full object-contain bg-gray-50" />
                <button
                  onClick={() => { setPreviewFile(null); setPreviewUrl(null); }}
                  className="absolute top-2 right-2 bg-white rounded-full p-1 shadow border border-gray-200 hover:bg-gray-100"
                >
                  <X size={14} />
                </button>
              </div>
            ) : (
              <div
                {...getRootProps()}
                className={clsx(
                  "w-full h-40 border-2 border-dashed rounded-xl flex flex-col items-center justify-center cursor-pointer transition-colors",
                  isDragActive
                    ? "border-indigo-400 bg-indigo-50"
                    : "border-gray-200 hover:border-indigo-300 hover:bg-gray-50"
                )}
              >
                <input {...getInputProps()} />
                <Upload size={24} className="text-gray-400 mb-2" />
                <p className="text-sm text-gray-500">
                  {isDragActive ? "Drop it here" : "Drag & drop or click to upload"}
                </p>
                <p className="text-xs text-gray-400 mt-1">JPEG, PNG, WEBP — max 5MB</p>
              </div>
            )}
          </div>
        )}

        {/* Category + Submit row */}
        <div className="flex gap-2">
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            className="border border-gray-200 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400 bg-white"
          >
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>

          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            className="flex-1 flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 text-white rounded-xl py-2 text-sm font-medium transition-colors"
          >
            {loading ? (
              <><Loader2 size={14} className="animate-spin" /> Searching...</>
            ) : (
              <><Search size={14} /> Search</>
            )}
          </button>

          <button
            onClick={handleReset}
            className="px-3 py-2 border border-gray-200 rounded-xl text-sm text-gray-500 hover:bg-gray-50 transition-colors"
          >
            <X size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
