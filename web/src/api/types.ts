export type ScoreTag = '高相似' | '中高相似' | '局部相似' | '差异明显' | '无有效评分'

export type CandidateCard = {
  candidate_id: string
  candidate_name: string
  candidate_path: string
  final_score: number
  contour_score: number | null
  part_score: number | null
  contour_diff_image: string | null
  analysis: string[]
}

export type CompareResponse = {
  run_id: string
  query_name: string
  query_staged_path: string
  results: CandidateCard[]
}

export type GalleryItem = {
  id: string
  name: string
  url: string
}

export type GalleryResponse = {
  total: number
  items: GalleryItem[]
}

export type PartEvidence = {
  part_name: string
  fused: number
  tag: ScoreTag
  tiles: {
    a_color: string
    b_color: string
    a_gray: string
    b_gray: string
  }
  metrics?: {
    clip?: number
    dino?: number
    ssim?: number
    edge?: number
  }
}

export type CandidateDetailResponse = {
  run_id: string
  query: {
    name: string
    image_url: string
    annotation_url: string
  }
  candidate: {
    id: string
    name: string
    image_url: string
    annotation_url: string
  }
  summary: {
    final_score: number
    tag: ScoreTag
    points: string[]
    weights: {
      contour: { weight: number; score: number }
      parts: { weight: number; score: number }
    }
  }
  contour: {
    score: number
    tag: ScoreTag
    diff_image_url: string
    conclusion: string
  }
  parts: {
    score: number
    tag: ScoreTag
    a_annotation_url: string
    b_annotation_url: string
    evidence: PartEvidence[]
  }
}

