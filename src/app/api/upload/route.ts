import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
    const formData = await req.formData();
    const file = formData.get("file") as File;

    if (!file) {
        return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    // ส่งไฟล์ไป Python ingest service
    const pyRes = await fetch(process.env.INGEST_API_URL + "/ingest", {
        method: "POST",
        body: formData,
    });

    if (!pyRes.ok) {
        return NextResponse.json({ error: "Ingest failed" }, { status: 500 });
    }

    return NextResponse.json({ status: "ok" });
}
