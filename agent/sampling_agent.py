"""
采样质量智能分析 Agent
基于规则推理的 DSP 实验助手 —— 不使用 LLM，纯规则引擎

推理层次:
  Layer 1: 奈奎斯特条件判断 (fs vs 2·fmax)
  Layer 2: 过采样/临界/欠采样分类
  Layer 3: FFT 混叠峰验证
  Layer 4: 综合评分 + 优化建议
"""

from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class SamplingAnalysis:
    """采样分析结果数据类"""
    nyquist_rate: float              # 所需奈奎斯特采样率 (Hz)
    nyquist_freq: float              # 奈奎斯特频率 fs/2 (Hz)
    fs: float                        # 当前采样率 (Hz)
    fmax: float                      # 信号最高频率分量 (Hz)
    ratio: float                     # 采样比 fs/fmax
    is_aliasing: bool                # 是否发生混叠
    is_critical: bool                # 是否临界采样
    is_oversampling: bool            # 是否过采样
    is_undersampling: bool           # 是否欠采样
    sampling_status: str             # 采样状态中文描述
    aliased_freqs: List[dict] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    summary_lines: List[str] = field(default_factory=list)


class SamplingAgent:
    """采样质量智能分析 Agent

    根据信号频率、采样率、混叠分析结果和重建误差，
    使用规则推理输出专业的 DSP 实验分析报告。

    使用示例:
        agent = SamplingAgent()
        analysis = agent.analyze(
            signal_freqs=[50, 120],
            fs=100,
            aliasing_result=aliasing.check_aliasing(...),
            noise_level=0.1,
            reconstruction_error=0.05,
        )
    """

    def analyze(self,
                signal_freqs: List[float],
                fs: float,
                aliasing_result: Optional[dict] = None,
                noise_level: float = 0.0,
                reconstruction_error: Optional[float] = None) -> SamplingAnalysis:
        """综合分析采样质量并生成报告

        Args:
            signal_freqs: 信号频率分量列表 (Hz)
            fs: 采样率 (Hz)
            aliasing_result: core.aliasing.check_aliasing() 的返回结果
            noise_level: 噪声标准差
            reconstruction_error: 重建均方误差 MSE

        Returns:
            SamplingAnalysis 对象，包含完整分析结果
        """
        # ---- 基础计算 ----
        fmax = max(signal_freqs) if signal_freqs else 0
        nyquist_rate = 2 * fmax
        nyquist_freq = fs / 2
        ratio = fs / fmax if fmax > 0 else float('inf')

        # ---- Layer 1: 奈奎斯特条件 ----
        is_aliasing = fs < nyquist_rate

        # ---- Layer 2: 采样状态分类 ----
        if ratio < 1.0:
            is_undersampling = True
            is_critical = False
            is_oversampling = False
        elif ratio < 2.0:
            is_undersampling = True
            is_critical = False
            is_oversampling = False
        elif ratio < 2.2:
            is_undersampling = False
            is_critical = True
            is_oversampling = False
        elif ratio < 5.0:
            is_undersampling = False
            is_critical = False
            is_oversampling = False
        else:
            is_undersampling = False
            is_critical = False
            is_oversampling = True

        # ---- 状态文本 ----
        if ratio < 1.0:
            sampling_status = f"严重欠采样（fs={fs:.0f} < fmax={fmax:.0f}），频谱已完全混叠"
        elif ratio < 2.0:
            sampling_status = f"欠采样（fs={fs:.0f} < 2·fmax={nyquist_rate:.0f}），不满足奈奎斯特定理"
        elif is_critical:
            sampling_status = "临界采样，接近奈奎斯特极限，理论上可重建但无余量"
        elif is_oversampling:
            sampling_status = f"深度过采样（fs/fmax={ratio:.1f}），信号可完美重建"
        elif ratio >= 3.0:
            sampling_status = f"过采样（fs/fmax={ratio:.1f}），重建质量良好"
        else:
            sampling_status = "正常采样，满足奈奎斯特定理，信号可以正确重建"

        # ---- Layer 3 & 4: 综合建议 ----
        suggestions = self._generate_suggestions(
            fs, fmax, nyquist_rate, nyquist_freq, ratio,
            is_aliasing, is_critical, is_oversampling,
            aliasing_result, noise_level, reconstruction_error,
        )

        # ---- 综合摘要 ----
        summary_lines = self._build_summary(
            fmax, nyquist_rate, fs, nyquist_freq, ratio,
            sampling_status, aliasing_result, signal_freqs,
        )

        alias_freqs = (aliasing_result.get('aliased_freqs', [])
                       if aliasing_result else [])

        return SamplingAnalysis(
            nyquist_rate=nyquist_rate,
            nyquist_freq=nyquist_freq,
            fs=fs,
            fmax=fmax,
            ratio=ratio,
            is_aliasing=is_aliasing,
            is_critical=is_critical,
            is_oversampling=is_oversampling,
            is_undersampling=is_undersampling,
            sampling_status=sampling_status,
            aliased_freqs=alias_freqs,
            suggestions=suggestions,
            summary_lines=summary_lines,
        )

    # ------------------------------------------------------------------
    # 规则引擎：建议生成
    # ------------------------------------------------------------------

    def _generate_suggestions(self,
                               fs: float,
                               fmax: float,
                               nyquist_rate: float,
                               nyquist_freq: float,
                               ratio: float,
                               is_aliasing: bool,
                               is_critical: bool,
                               is_oversampling: bool,
                               aliasing_result: Optional[dict],
                               noise_level: float,
                               reconstruction_error: Optional[float],
                               ) -> List[str]:
        """基于规则的建议生成器"""
        suggestions: List[str] = []

        # --- 混叠相关建议 ---
        if is_aliasing:
            suggestions.append(
                f"提高采样率至 {nyquist_rate:.0f} Hz 以上"
                f"（奈奎斯特率 = 2 × {fmax:.0f} Hz = {nyquist_rate:.0f} Hz）"
            )
            suggestions.append(
                "在实际系统中应使用抗混叠低通滤波器，"
                f"截止频率设为 {nyquist_freq:.0f} Hz 以下，滤除高于 fs/2 的频率分量"
            )

            # 逐分量说明混叠情况
            if aliasing_result and aliasing_result.get('aliased_freqs'):
                for af in aliasing_result['aliased_freqs']:
                    f_orig = af['original']
                    f_alias = af['alias']
                    suggestions.append(
                        f"频率 {f_orig:.0f} Hz → 混叠为 {f_alias:.1f} Hz，"
                        f"重建后将在 {f_alias:.1f} Hz 处出现虚假低频分量"
                    )

        elif is_critical:
            suggestions.append(
                f"当前处于临界采样状态（fs/fmax={ratio:.2f}），"
                f"建议提高采样率至 {nyquist_rate * 1.5:.0f} Hz 以上，"
                "为实际抗混叠滤波器留出过渡带"
            )
        elif is_oversampling:
            if ratio > 10:
                suggestions.append(
                    "采样率远高于需求，可以考虑适当降低以减少数据量和处理开销，"
                    f"但需保持 fs ≥ {nyquist_rate:.0f} Hz"
                )
            else:
                suggestions.append(
                    "当前采样率充足，信号各频率分量均可正确采集和重建"
                )

        # --- 噪声相关建议 ---
        if noise_level > 0.5:
            suggestions.append(
                f"噪声水平较高（σ={noise_level:.2f}），"
                "建议在采样前对信号进行模拟预滤波以抑制噪声"
            )
        elif noise_level > 0.2:
            suggestions.append(
                f"存在中等强度噪声（σ={noise_level:.2f}），"
                "可通过提高采样率并做数字滤波来改善信噪比"
            )

        # --- 重建误差相关 ---
        if reconstruction_error is not None:
            if reconstruction_error > 0.5:
                suggestions.append(
                    f"重建误差很大（MSE={reconstruction_error:.4f}），"
                    "重建信号已严重失真，主要原因可能是采样率不足导致的混叠"
                )
            elif reconstruction_error > 0.05:
                suggestions.append(
                    f"重建误差中等（MSE={reconstruction_error:.4f}），"
                    "可尝试提高采样率或增大 sinc 插值窗口以改善重建质量"
                )

        # --- 如果没有问题 ---
        if not suggestions:
            suggestions.append("当前采样设置合理，信号可以高质量重建，无需调整")

        return suggestions

    # ------------------------------------------------------------------
    # 摘要生成
    # ------------------------------------------------------------------

    def _build_summary(self,
                        fmax: float,
                        nyquist_rate: float,
                        fs: float,
                        nyquist_freq: float,
                        ratio: float,
                        sampling_status: str,
                        aliasing_result: Optional[dict],
                        signal_freqs: List[float],
                        ) -> List[str]:
        """构建结构化摘要"""
        lines = [
            f"信号最高频率 fmax = {fmax:.1f} Hz",
            f"信号频率分量: {', '.join(f'{f:.0f} Hz' for f in signal_freqs)}",
            f"奈奎斯特采样率 = 2 × {fmax:.0f} = {nyquist_rate:.1f} Hz",
            f"当前采样率 fs = {fs:.1f} Hz",
            f"奈奎斯特频率 fs/2 = {nyquist_freq:.1f} Hz",
            f"采样比 fs/fmax = {ratio:.2f}",
            f"状态判定: {sampling_status}",
        ]

        if aliasing_result and aliasing_result.get('aliased_freqs'):
            lines.append(f"混叠分量数: {len(aliasing_result['aliased_freqs'])}")
            for af in aliasing_result['aliased_freqs']:
                lines.append(
                    f"  └ {af['original']:.0f} Hz → {af['alias']:.1f} Hz (频率折叠)"
                )

        return lines
