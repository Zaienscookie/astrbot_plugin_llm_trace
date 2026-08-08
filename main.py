"""
llm_trace v1.1 - LLM 调用溯源（text_chat 接管版）

原理：直接包装 provider 实例的 text_chat 方法，因此无论调用来自 AstrBot 主 agent、
双子插件还是其他插件，每次 LLM 调用都能被记录。

每次调用成功后，在 AstrBot 日志输出一行：
  [LLM-Trace] session=xxx | 供应商=xxx | 模型=xxx | API key=前8位 | 耗时=x.xs
调用失败则输出 warning。

不往群里发任何消息，纯粹记录日志。
"""

import time

from astrbot import logger
from astrbot.api.event import AstrMessageEvent, filter as filter_mod
from astrbot.api.star import Context, Star, register


@register("astrbot_plugin_llm_trace", "扎恩斯", "LLM 调用溯源：日志记录实际供应商/模型/API key", "1.1.0")
class LLMTracePlugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}
        self._wrapped = set()  # 已包装的 provider 实例 id
        # 插件加载时若 provider 已就绪则立即接管（覆盖热重载场景）
        try:
            self._wrap_all()
        except Exception:  # noqa: BLE001
            pass

    @filter_mod.on_astrbot_loaded()
    async def on_astrbot_loaded(self, event: AstrMessageEvent | None = None):
        """AstrBot 加载完成后，接管所有已加载的 chat provider"""
        self._wrap_all()

    def _wrap_all(self):
        try:
            pm = self.context.provider_manager
            insts = list(getattr(pm, "provider_insts", []) or [])
            for inst in insts:
                self._wrap_one(inst)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[LLM-Trace] 接管 provider 失败: {type(e).__name__}: {e}")

    def _wrap_one(self, inst):
        """把实例的 text_chat 包一层记录日志（幂等）"""
        if getattr(inst, "_llm_trace_wrapped", False) or not hasattr(inst, "text_chat"):
            return
        orig = inst.text_chat
        pid = ((inst.provider_config or {}).get("id")) or "?"

        async def wrapped(*args, _orig=orig, _inst=inst, **kwargs):
            t0 = time.time()
            try:
                ret = await _orig(*args, **kwargs)
                self._log(_inst, kwargs, ok=True, secs=time.time() - t0)
                return ret
            except Exception as e:  # noqa: BLE001
                self._log(_inst, kwargs, ok=False, err=e, secs=time.time() - t0)
                raise

        inst.text_chat = wrapped
        inst._llm_trace_wrapped = True  # 实例级标记：插件重载也不重复包装
        self._wrapped.add(id(inst))
        logger.info(f"[LLM-Trace] 已接管 provider: {pid}")

    def _log(self, inst, kwargs, ok: bool, err=None, secs: float = 0.0):
        try:
            pc = getattr(inst, "provider_config", {}) or {}
            pid = pc.get("id", "?")
            model = inst.get_model() or pc.get("model", "?")
            keys = pc.get("key") or []
            key8 = (str(keys[0])[:8] + "****") if keys else "无"
            sid = kwargs.get("session_id", "?")
            if ok:
                logger.info(
                    f"[LLM-Trace] session={sid} | 供应商={pid} | "
                    f"模型={model} | API key={key8} | 耗时={secs:.1f}s"
                )
            else:
                logger.warning(
                    f"[LLM-Trace] session={sid} | 供应商={pid} | 模型={model} | "
                    f"调用失败: {type(err).__name__}: {err}"
                )
        except Exception:  # noqa: BLE001
            # 记录日志本身绝不能再抛异常
            pass
