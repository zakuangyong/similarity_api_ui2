import type {
  CandidateCard,
  CandidateDetailResponse,
  CompareResponse,
  GalleryItem,
  GalleryResponse,
  ScoreTag,
} from './types'

type Car = { id: string; name: string; url: string }

const cars: Car[] = [
  { id: 'porsche-911', name: '保时捷911', url: '/img/front/保时捷911.svg' },
  { id: 'buick-gl8', name: '别克GL8', url: '/img/front/别克GL8.svg' },
  { id: 'benz-c', name: '奔驰C级', url: '/img/front/奔驰C级.svg' },
  { id: 'audi-a4l', name: '奥迪A4L', url: '/img/front/奥迪A4L.svg' },
  { id: 'audi-a6l', name: '奥迪A6L', url: '/img/front/奥迪A6L.svg' },
  { id: 'audi-q5l', name: '奥迪Q5L', url: '/img/front/奥迪Q5L.svg' },
  { id: 'bmw-3', name: '宝马3系', url: '/img/front/宝马3系.svg' },
  { id: 'bmw-5', name: '宝马5系', url: '/img/front/宝马5系.svg' },
  { id: 'xiaomi-su7', name: '小米su7', url: '/img/front/小米su7.svg' },
  { id: 'xingyao-7', name: '星耀7', url: '/img/front/星耀7.svg' },
  { id: 'han', name: '汉', url: '/img/front/汉.svg' },
  { id: 'haibao-06', name: '海豹06', url: '/img/front/海豹06.svg' },
  { id: 'hongqi-h5', name: '红旗H5', url: '/img/front/红旗H5.svg' },
  { id: 'omoda', name: '欧萌达', url: '/img/front/欧萌达.svg' },
]

function toTag(score: number): ScoreTag {
  if (!Number.isFinite(score)) return '无有效评分'
  if (score >= 85) return '高相似'
  if (score >= 72) return '中高相似'
  if (score >= 65) return '局部相似'
  return '差异明显'
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms))
}

function defaultAnalysis(final: number, contour: number): string[] {
  const tag = toTag(final)
  const points: string[] = []
  points.push(`最终评分 ${final.toFixed(1)} 分，判定为${tag}，适合进入人工复核。`)
  points.push(`整车轮廓相似度 ${contour.toFixed(1)}，车身基础比例和正面姿态高度接近。`)
  points.push(`共对比 6 个有效部件，后视镜相似度最高，前挡风玻璃差异较明显。`)
  return points
}

export async function mockCompare(topk: number): Promise<CompareResponse> {
  await sleep(260)

  const runId = `run_${Date.now().toString(36)}`
  const ordered: Array<{ car: Car; final: number }> = [
    { car: cars.find((c) => c.id === 'hongqi-h5')!, final: 90.0 },
    { car: cars.find((c) => c.id === 'haibao-06')!, final: 78.8 },
    { car: cars.find((c) => c.id === 'bmw-3')!, final: 76.6 },
    { car: cars.find((c) => c.id === 'audi-q5l')!, final: 75.2 },
    { car: cars.find((c) => c.id === 'buick-gl8')!, final: 73.4 },
    { car: cars.find((c) => c.id === 'porsche-911')!, final: 70.8 },
    { car: cars.find((c) => c.id === 'audi-a6l')!, final: 69.4 },
    { car: cars.find((c) => c.id === 'han')!, final: 68.7 },
    { car: cars.find((c) => c.id === 'bmw-5')!, final: 67.2 },
    { car: cars.find((c) => c.id === 'benz-c')!, final: 65.9 },
  ]

  const results: CandidateCard[] = ordered.slice(0, Math.max(1, topk)).map(({ car, final }, idx) => {
    const contour = idx === 1 ? 90.6 : Math.min(95, final + 10)
    const parts = final
    return {
      candidate_id: car.id,
      candidate_name: car.name,
      candidate_path: car.url,
      final_score: final,
      contour_score: contour,
      part_score: parts,
      contour_diff_image: '/prototype-assets/edge_diff.jpg',
      analysis: defaultAnalysis(final, contour),
    }
  })

  return {
    run_id: runId,
    query_name: '上传比对图片',
    query_staged_path: '/img/front/星耀7.svg',
    results,
  }
}

export async function mockGetGallery(): Promise<GalleryResponse> {
  await sleep(140)
  const items: GalleryItem[] = cars.map((c) => ({ id: c.id, name: c.name, url: c.url }))
  return { total: items.length, items }
}

export async function mockGetCandidateDetail(runId: string, candidateId: string): Promise<CandidateDetailResponse> {
  await sleep(180)
  const candidate = cars.find((c) => c.id === candidateId) ?? cars[0]
  const finalScore = candidateId === 'haibao-06' ? 78.8 : 78.8
  const contourScore = 90.6
  const partScore = 78.8

  return {
    run_id: runId,
    query: {
      name: 'A车',
      image_url: '/img/front/星耀7.svg',
      annotation_url: '/prototype-assets/A_annotation.jpg',
    },
    candidate: {
      id: candidate.id,
      name: 'B车',
      image_url: candidate.url,
      annotation_url: '/prototype-assets/B_annotation.jpg',
    },
    summary: {
      final_score: finalScore,
      tag: toTag(finalScore),
      points: defaultAnalysis(finalScore, contourScore),
      weights: {
        contour: { weight: 0.4, score: contourScore },
        parts: { weight: 0.6, score: partScore },
      },
    },
    contour: {
      score: contourScore,
      tag: toTag(contourScore),
      diff_image_url: '/prototype-assets/edge_diff.jpg',
      conclusion: '结论：两车正面主体轮廓接近，差异主要集中在车头下沿、车顶线和局部外扩区域。',
    },
    parts: {
      score: partScore,
      tag: toTag(partScore),
      a_annotation_url: '/prototype-assets/A_annotation.jpg',
      b_annotation_url: '/prototype-assets/B_annotation.jpg',
      evidence: [
        {
          part_name: '后视镜',
          fused: 96.8,
          tag: '高相似',
          tiles: {
            a_color: '/img/front/星耀7.svg',
            b_color: candidate.url,
            a_gray: '/img/front/星耀7.svg',
            b_gray: candidate.url,
            diff: '',
          },
        },
        {
          part_name: '车灯',
          fused: 86.4,
          tag: '高相似',
          tiles: {
            a_color: '/img/front/星耀7.svg',
            b_color: candidate.url,
            a_gray: '/img/front/星耀7.svg',
            b_gray: candidate.url,
            diff: '',
          },
        },
        {
          part_name: '前保险杠',
          fused: 81.6,
          tag: '中高相似',
          tiles: {
            a_color: '/img/front/星耀7.svg',
            b_color: candidate.url,
            a_gray: '/img/front/星耀7.svg',
            b_gray: candidate.url,
            diff: '',
          },
        },
        {
          part_name: '前挡风玻璃',
          fused: 64.6,
          tag: '差异明显',
          tiles: {
            a_color: '/img/front/星耀7.svg',
            b_color: candidate.url,
            a_gray: '/img/front/星耀7.svg',
            b_gray: candidate.url,
            diff: '',
          },
        },
      ],
    },
  }
}
