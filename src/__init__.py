"""
DermoAI Pipeline — Dermatological lesion classification and reporting system.

Components:
    - MelanomaClassifier: EfficientNet-B0 classifier (fp16, CUDA)
    - GradCAMExplainer: GradCAM++ visual explainability
    - ABCDEReportGenerator: BioMistral-7B ABCDE report generation
    - ReportEvaluator: NLG metrics (ROUGE, BLEU, BERTScore)
    - QualitativeEvaluator: Rubric-based qualitative evaluation (CPU)
    - CrossModelEvaluator: Cross-model comparison without ground truth
    - RAGRetriever: PubMed + DermNet knowledge base retrieval
    - DermAIPipeline: End-to-end orchestration with VRAM management
"""

__version__ = "1.0.0"
__author__ = "DermoAI Pipeline"
