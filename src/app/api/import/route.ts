import { db, sqlite } from "@/db";
import { importBatches, arRecords } from "@/db/schema";
import { eq } from "drizzle-orm";
import { NextResponse } from "next/server";
import { v4 as uuid } from "uuid";
import { safeJsonParse } from "@/lib/utils";
import { config } from "@/lib/config";
import { handleApiError } from "@/lib/api-error";
import * as XLSX from "xlsx";

const COLUMN_VARIANTS: Record<string, string[]> = {
  customerName: ["客户名称", "客户", "单位名称", "单位", "名称", "对方单位", "往来单位", "customer_name", "customer"],
  customerCode: ["客户编码", "客户代码", "编码", "代码", "customer_code", "code"],
  totalBalance: ["余额", "期末余额", "账面余额", "总余额", "应收账款余额", "金额", "balance", "total_balance", "amount"],
  within1Year: ["1年以内", "一年以内", "1年内", "within_1_year"],
  year1to2: ["1-2年", "1至2年", "一到两年", "year_1_to_2"],
  year2to3: ["2-3年", "2至3年", "两到三年", "year_2_to_3"],
  year3to4: ["3-4年", "3至4年", "三到四年", "year_3_to_4"],
  year4to5: ["4-5年", "4至5年", "四到五年", "year_4_to_5"],
  over5Years: ["5年以上", "五年以上", "over_5_years"],
  isRelatedParty: ["关联方", "是否关联方", "related_party"],
  contactPerson: ["联系人", "联系人姓名", "负责人"],
  contactPhone: ["联系电话", "电话", "手机"],
  contactEmail: ["邮箱", "电子邮箱", "email"],
  address: ["地址", "通讯地址", "邮寄地址", "联系地址"],
  city: ["城市", "所在城市"],
  province: ["省份", "省", "所在省份"],
  postalCode: ["邮编", "邮政编码"],
};

function autoMapColumns(headers: string[]): Record<string, string> {
  const mapping: Record<string, string> = {};
  const usedFields = new Set<string>();

  // First pass: exact matches
  for (const header of headers) {
    const trimmed = header.trim();
    for (const [field, variants] of Object.entries(COLUMN_VARIANTS)) {
      if (usedFields.has(field)) continue;
      if (variants.some((v) => v === trimmed)) {
        mapping[trimmed] = field;
        usedFields.add(field);
        break;
      }
    }
  }

  // Second pass: fuzzy matches for unmapped headers
  for (const header of headers) {
    const trimmed = header.trim();
    if (mapping[trimmed]) continue;
    for (const [field, variants] of Object.entries(COLUMN_VARIANTS)) {
      if (usedFields.has(field)) continue;
      if (variants.some((v) => trimmed.includes(v) || v.includes(trimmed))) {
        mapping[trimmed] = field;
        usedFields.add(field);
        break;
      }
    }
  }

  return mapping;
}

function classifyRisk(record: any): "low" | "medium" | "high" {
  const balance = record.totalBalance || 0;
  const longAged = (record.year2to3 || 0) + (record.year3to4 || 0) + (record.year4to5 || 0) + (record.over5Years || 0);
  if (balance > config.riskHighBalanceThreshold || longAged > balance * config.riskHighAgingRatio) return "high";
  if (balance > config.riskMediumBalanceThreshold || longAged > balance * config.riskMediumAgingRatio) return "medium";
  return "low";
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;
    const projectId = formData.get("projectId") as string;
    const columnMappingStr = formData.get("columnMapping") as string | null;

    if (!file || !projectId) {
      return NextResponse.json({ error: "文件和项目ID不能为空" }, { status: 400 });
    }

    if (file.size > config.maxImportFileSize) {
      return NextResponse.json({ error: "文件大小不能超过10MB" }, { status: 400 });
    }

    const buffer = Buffer.from(await file.arrayBuffer());
    const workbook = XLSX.read(buffer, { type: "buffer" });
    if (!workbook.SheetNames.length) {
      return NextResponse.json({ error: "文件中没有工作表" }, { status: 400 });
    }
    const sheetName = workbook.SheetNames[0];
    const sheet = workbook.Sheets[sheetName];
    const rawData = XLSX.utils.sheet_to_json<Record<string, any>>(sheet);

    if (rawData.length === 0) {
      return NextResponse.json({ error: "文件中没有数据" }, { status: 400 });
    }

    const headers = Object.keys(rawData[0]);
    const columnMapping = columnMappingStr
      ? safeJsonParse(columnMappingStr, autoMapColumns(headers))
      : autoMapColumns(headers);

    // If this is a preview request
    const preview = formData.get("preview");
    if (preview === "true") {
      return NextResponse.json({
        headers,
        columnMapping,
        previewRows: rawData.slice(0, 5),
        totalRows: rawData.length,
      });
    }

    const batchId = uuid();
    const now = new Date().toISOString();
    const headerToField: Record<string, string> = columnMapping;

    const insertedCount = sqlite.transaction(() => {
      db.insert(importBatches).values({
        id: batchId,
        projectId,
        fileName: file.name,
        recordCount: rawData.length,
        columnMapping: JSON.stringify(columnMapping),
        status: "processing",
        createdAt: now,
      }).run();

      let count = 0;
      for (const row of rawData) {
        const mapped: any = {};
        for (const [header, value] of Object.entries(row)) {
          const field = headerToField[header];
          if (field) {
            mapped[field] = value;
          }
        }

        if (!mapped.customerName) continue;

        const totalBalance = Number(mapped.totalBalance) || 0;
        const record = {
          customerName: String(mapped.customerName),
          customerCode: mapped.customerCode ? String(mapped.customerCode) : null,
          totalBalance,
          within1Year: Number(mapped.within1Year) || 0,
          year1to2: Number(mapped.year1to2) || 0,
          year2to3: Number(mapped.year2to3) || 0,
          year3to4: Number(mapped.year3to4) || 0,
          year4to5: Number(mapped.year4to5) || 0,
          over5Years: Number(mapped.over5Years) || 0,
          isRelatedParty: mapped.isRelatedParty === "是" || mapped.isRelatedParty === true || mapped.isRelatedParty === 1,
          contactPerson: mapped.contactPerson ? String(mapped.contactPerson) : null,
          contactPhone: mapped.contactPhone ? String(mapped.contactPhone) : null,
          contactEmail: mapped.contactEmail ? String(mapped.contactEmail) : null,
          address: mapped.address ? String(mapped.address) : null,
          city: mapped.city ? String(mapped.city) : null,
          province: mapped.province ? String(mapped.province) : null,
          postalCode: mapped.postalCode ? String(mapped.postalCode) : null,
        };

        const riskLevel = classifyRisk(record);

        db.insert(arRecords).values({
          id: uuid(),
          projectId,
          importBatchId: batchId,
          ...record,
          riskLevel,
          selectionStatus: "unselected",
          createdAt: now,
        }).run();
        count++;
      }

      db.update(importBatches)
        .set({ status: "completed", recordCount: count })
        .where(eq(importBatches.id, batchId))
        .run();

      return count;
    })();

    return NextResponse.json({
      batchId,
      insertedCount,
      skippedCount: rawData.length - insertedCount,
      totalRows: rawData.length,
      columnMapping,
    }, { status: 201 });
  } catch (error) {
    return handleApiError(error, "导入失败");
  }
}
