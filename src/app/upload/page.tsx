"use client";
import { useState } from "react";

export default function UploadPage() {
    const [file, setFile] = useState<File | null>(null);
    const [status, setStatus] = useState<string>("");

    const handleUpload = async () => {
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        setStatus("Uploading...");

        const res = await fetch("/api/upload", {
            method: "POST",
            body: formData,
        });

        if (res.ok) {
            setStatus("✅ Upload success! Ingest started.");
        } else {
            setStatus("❌ Upload failed.");
        }
    };

    return (
        <div className="flex flex-col items-center justify-center h-screen gap-6">
            <h1 className="text-2xl font-bold">📄 Upload PDF to RAG</h1>

            <input
                type="file"
                accept="application/pdf"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="border p-2 rounded"
            />

            <button
                onClick={handleUpload}
                disabled={!file}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg disabled:opacity-50"
            >
                Upload
            </button>

            {status && <p>{status}</p>}
        </div>
    );
}
