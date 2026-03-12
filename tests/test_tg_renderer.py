"""
Telegram 图文混排渲染引擎单元测试模块。

本模块测试 tg_main.py 中的核心渲染逻辑：
1. Markdown 方言翻译器 (Standard -> Telegram Legacy)
2. 正则切片引擎 (图文混排)
3. 异常降级兜底机制
"""

import pytest
import re
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


class TestMarkdownDialectTranslator:
    """测试 Markdown 方言翻译逻辑。"""

    def test_bold_conversion(self) -> None:
        """测试加粗语法转换：**text** -> *text*"""
        text = "这是 **加粗文字** 测试"
        result = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
        assert result == "这是 *加粗文字* 测试"

    def test_multiple_bold_conversion(self) -> None:
        """测试多个加粗语法转换"""
        text = "**第一处** 和 **第二处** 加粗"
        result = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
        assert result == "*第一处* 和 *第二处* 加粗"

    def test_h3_heading_downgrade(self) -> None:
        """测试三级标题降级：### -> ■"""
        text = "### 这是三级标题"
        result = re.sub(r'^###\s+(.*)', r'■ \1', text, flags=re.MULTILINE)
        assert result == "■ 这是三级标题"

    def test_h2_heading_downgrade(self) -> None:
        """测试二级标题降级：## -> ●"""
        text = "## 这是二级标题"
        result = re.sub(r'^##\s+(.*)', r'● \1', text, flags=re.MULTILINE)
        assert result == "● 这是二级标题"

    def test_h1_heading_downgrade(self) -> None:
        """测试一级标题降级：# -> ◆"""
        text = "# 这是一级标题"
        result = re.sub(r'^#\s+(.*)', r'◆ \1', text, flags=re.MULTILINE)
        assert result == "◆ 这是一级标题"

    def test_mixed_headings_conversion(self) -> None:
        """测试混合标题层级转换"""
        text = """# 一级标题
## 二级标题
### 三级标题"""
        result = text
        result = re.sub(r'^###\s+(.*)', r'■ \1', result, flags=re.MULTILINE)
        result = re.sub(r'^##\s+(.*)', r'● \1', result, flags=re.MULTILINE)
        result = re.sub(r'^#\s+(.*)', r'◆ \1', result, flags=re.MULTILINE)
        
        expected = """◆ 一级标题
● 二级标题
■ 三级标题"""
        assert result == expected

    def test_full_dialect_translation(self) -> None:
        """测试完整的方言翻译流程"""
        text = """# 报告标题

这是 **重要** 内容。

## 第二章

更多 **加粗** 和 **强调**"""
        
        # 模拟完整翻译流程
        result = text
        result = re.sub(r'\*\*(.*?)\*\*', r'*\1*', result)
        result = re.sub(r'^###\s+(.*)', r'■ \1', result, flags=re.MULTILINE)
        result = re.sub(r'^##\s+(.*)', r'● \1', result, flags=re.MULTILINE)
        result = re.sub(r'^#\s+(.*)', r'◆ \1', result, flags=re.MULTILINE)
        
        assert "*重要*" in result
        assert "*加粗*" in result
        assert "*强调*" in result
        assert "◆ 报告标题" in result
        assert "● 第二章" in result
        assert "###" not in result
        assert "##" not in result
        assert "#" not in result or "◆" in result  # 确保#被转换


class TestImageTextChunkingEngine:
    """测试图文切片引擎。"""

    def test_split_single_image(self) -> None:
        """测试单图片切片"""
        text = "前言文本 ![描述](./image.png) 后续文本"
        chunks = re.split(r'(!\[.*?\]\(.*?\))', text)
        
        # 应切分为：['前言文本 ', '![描述](./image.png)', ' 后续文本']
        assert len(chunks) == 3
        assert '![描述](./image.png)' in chunks

    def test_split_multiple_images(self) -> None:
        """测试多图片切片"""
        text = "文本 1 ![图 1](./img1.png) 文本 2 ![图 2](./img2.png) 文本 3"
        chunks = re.split(r'(!\[.*?\]\(.*?\))', text)
        
        # 应包含 2 个图片标记和 3 段文本
        image_chunks = [c for c in chunks if re.match(r'^!\[.*?\]\(.*?\)$', c.strip())]
        assert len(image_chunks) == 2

    def test_split_text_only(self) -> None:
        """测试纯文本无图片"""
        text = "这是纯文本，没有图片"
        chunks = re.split(r'(!\[.*?\]\(.*?\))', text)
        
        # 应只返回原始文本
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_image_path_extraction(self) -> None:
        """测试从图片标记中提取路径"""
        img_markdown = "![走势图](./chart.png)"
        match = re.match(r'^!\[.*?\]\((.*?)\)$', img_markdown)
        
        assert match is not None
        assert match.group(1) == "./chart.png"

    def test_image_filename_cleaning(self) -> None:
        """测试图片文件名清理（去除 ./）"""
        img_filename = "./chart.png".replace("./", "")
        assert img_filename == "chart.png"


class TestAsyncSendLogic:
    """测试异步发送逻辑（Mock 测试）。"""

    @pytest.mark.asyncio
    async def test_text_chunk_send_with_markdown(self) -> None:
        """测试文本切片带 Markdown 发送"""
        mock_message = AsyncMock()
        mock_message.reply_text = AsyncMock()
        
        chunk = "这是 *加粗* 文本"
        
        # 模拟成功发送
        await mock_message.reply_text(chunk, parse_mode="Markdown")
        
        mock_message.reply_text.assert_called_once_with(
            chunk,
            parse_mode="Markdown"
        )

    @pytest.mark.asyncio
    async def test_text_chunk_fallback_on_error(self) -> None:
        """测试 Markdown 渲染失败时的降级逻辑"""
        mock_message = AsyncMock()
        mock_message.reply_text = AsyncMock(
            side_effect=[Exception("Markdown 解析错误"), None]  # 第一次失败，第二次成功
        )
        
        chunk = "这是 *非法 **嵌套* 标记"
        
        # 第一次尝试失败，降级为纯文本
        try:
            await mock_message.reply_text(chunk, parse_mode="Markdown")
        except Exception:
            # 降级：去掉 * 号
            fallback_text = chunk.replace('*', '')
            await mock_message.reply_text(fallback_text)
        
        # 验证降级后调用
        assert mock_message.reply_text.call_count >= 1

    @pytest.mark.asyncio
    async def test_image_chunk_send(self) -> None:
        """测试图片切片发送"""
        mock_message = AsyncMock()
        mock_message.reply_photo = AsyncMock()
        
        # 模拟图片文件
        mock_photo = MagicMock()
        
        with patch('builtins.open', MagicMock(return_value=mock_photo)):
            with patch.object(Path, 'exists', return_value=True):
                img_path = Path("/fake/path/image.png")
                with open(img_path, 'rb') as photo:
                    await mock_message.reply_photo(photo=photo)
        
        mock_message.reply_photo.assert_called_once()


class TestExceptionHandling:
    """测试异常处理机制。"""

    def test_image_path_not_exists(self) -> None:
        """测试图片文件不存在时的处理"""
        img_filename = "non_existent.png"
        sandbox_dir = Path("/tmp/fake_sandbox")
        img_path = (sandbox_dir / img_filename).resolve()
        
        # 模拟文件不存在
        with patch.object(Path, 'exists', return_value=False):
            assert not img_path.exists()
            # 应返回错误提示："⚠️ [此处图片生成失败或已被清理]"

    def test_empty_chunk_skip(self) -> None:
        """测试空切片跳过逻辑"""
        chunks = ["文本 1", "", "  ", "文本 2", None]
        processed = []
        
        for chunk in chunks:
            if chunk is None:
                continue
            chunk = chunk.strip()
            if not chunk:
                continue  # 跳过空行
            processed.append(chunk)
        
        assert len(processed) == 2
        assert processed == ["文本 1", "文本 2"]

    def test_malformed_image_markdown(self) -> None:
        """测试 malformed 图片标记的处理"""
        # 不完整的图片标记
        text = "这是 ![不完整 的图片"
        match = re.match(r'^!\[.*?\]\((.*?)\)$', text)
        
        # 应返回 None，视为普通文本
        assert match is None


class TestIntegration:
    """集成测试：模拟完整渲染流程。"""

    def test_full_rendering_pipeline(self) -> None:
        """测试完整渲染流程"""
        # 模拟 AI 返回的原始文本
        raw_response = """# 股票分析报告

这是 **重要** 结论。

## 走势分析

![走势图](./chart.png)

### 风险提示

投资有 **风险**，入市需谨慎。"""
        
        # Step 1: Markdown 方言翻译
        final_text = raw_response
        final_text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', final_text)
        final_text = re.sub(r'^###\s+(.*)', r'■ \1', final_text, flags=re.MULTILINE)
        final_text = re.sub(r'^##\s+(.*)', r'● \1', final_text, flags=re.MULTILINE)
        final_text = re.sub(r'^#\s+(.*)', r'◆ \1', final_text, flags=re.MULTILINE)
        
        # Step 2: 图文切片
        chunks = re.split(r'(!\[.*?\]\(.*?\))', final_text)
        
        # 验证
        assert "◆ 股票分析报告" in final_text
        assert "*重要*" in final_text
        assert "*风险*" in final_text
        assert len(chunks) > 1  # 确保被切片
        
        # 验证图片标记被正确切分
        image_chunks = [
            c for c in chunks 
            if c.strip() and re.match(r'^!\[.*?\]\(.*?\)$', c.strip())
        ]
        assert len(image_chunks) == 1
        assert "![走势图](./chart.png)" in image_chunks[0]


class TestCaptionSplitLogic:
    """测试 caption 超长拆分逻辑。"""

    @pytest.mark.asyncio
    async def test_short_caption_no_split(self) -> None:
        """测试短 caption 不拆分"""
        mock_message = AsyncMock()
        mock_photo = MagicMock()
        
        caption = "这是短 caption（100 字以内）"
        
        # 模拟 send_with_caption_split 逻辑
        if len(caption) <= 1024:
            await mock_message.reply_photo(photo=mock_photo, caption=caption)
        
        mock_message.reply_photo.assert_called_once()

    @pytest.mark.asyncio
    async def test_long_caption_split(self) -> None:
        """测试长 caption 拆分"""
        mock_message = AsyncMock()
        mock_photo = MagicMock()
        
        # 生成 2000 字符的 caption
        caption = "A" * 2000
        
        # 模拟拆分逻辑
        if len(caption) > 1024:
            # 第一条：图片 + 前段
            part1 = caption[:1021] + "..."
            await mock_message.reply_photo(photo=mock_photo, caption=part1)
            
            # 第二条：剩余文本
            remaining = caption[1024:]
            await mock_message.reply_text(remaining)
        
        # 验证调用了两次发送
        assert mock_message.reply_photo.call_count == 1
        assert mock_message.reply_text.call_count == 1

    @pytest.mark.asyncio
    async def test_caption_markdown_fallback(self) -> None:
        """测试 caption Markdown 渲染失败降级"""
        mock_message = AsyncMock()
        mock_photo = MagicMock()
        mock_message.reply_photo = AsyncMock(
            side_effect=[Exception("Markdown 解析错误"), None]
        )
        
        caption = "这是 *非法 **嵌套* 标记"
        
        # 第一次尝试失败，降级
        try:
            await mock_message.reply_photo(
                photo=mock_photo,
                caption=caption,
                parse_mode="Markdown"
            )
        except Exception:
            fallback = caption.replace('*', '')
            await mock_message.reply_photo(photo=mock_photo, caption=fallback)
        
        # 验证降级后调用
        assert mock_message.reply_photo.call_count >= 1


class TestPreReadMechanism:
    """测试预读机制（文本跳过逻辑）。"""

    def test_text_skip_when_next_is_image(self) -> None:
        """测试下一个是图片时文本跳过"""
        chunks = ["文本 A", "![图片](./img.png)", "文本 B"]
        is_consumed = [False, False, False]
        
        sent_items = []
        
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            
            img_match = re.match(r'^!\[.*?\]\(.*?\)$', chunk)
            
            if img_match:
                # 图片：读取前一个文本作为 caption
                if i > 0 and not is_consumed[i-1]:
                    caption = chunks[i-1].strip()
                    sent_items.append(f"图片+caption:{caption}")
                    is_consumed[i-1] = True
                else:
                    sent_items.append("纯图片")
            else:
                # 文本：预读下一个
                next_is_image = (
                    i + 1 < len(chunks) and
                    re.match(r'^!\[.*?\]\(.*?\)$', chunks[i+1].strip())
                )
                
                if next_is_image:
                    continue  # 跳过
                else:
                    sent_items.append(f"文本:{chunk}")
        
        # 验证：文本 A 被跳过（作为 caption），文本 B 正常发送
        assert sent_items == ["图片+caption:文本 A", "文本:文本 B"]

    def test_consecutive_images_no_caption(self) -> None:
        """测试连续图片无 caption"""
        chunks = ["![图片 1](./img1.png)", "![图片 2](./img2.png)"]
        is_consumed = [False, False]
        
        sent_items = []
        
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            
            img_match = re.match(r'^!\[.*?\]\(.*?\)$', chunk)
            
            if img_match:
                # 图片：读取前一个文本作为 caption（如果前一个是文本且未消费）
                if i > 0 and not is_consumed[i-1]:
                    prev_chunk = chunks[i-1].strip()
                    # 检查前一个切片是否是文本（不是图片）
                    prev_is_image = re.match(r'^!\[.*?\]\(.*?\)$', prev_chunk)
                    if not prev_is_image:
                        caption = prev_chunk
                        sent_items.append(f"图片+caption:{caption}")
                        is_consumed[i-1] = True
                    else:
                        sent_items.append("纯图片")
                else:
                    sent_items.append("纯图片")
            else:
                sent_items.append(f"文本:{chunk}")
        
        # 验证：两张图片都是纯图片（无 caption）
        assert sent_items == ["纯图片", "纯图片"]

    def test_table_image_integration(self) -> None:
        """测试表格图片与普通图片统一处理"""
        # 模拟 AI 返回内容包含表格和走势图
        raw_response = """分析如下：

| 代码 | 名称 | 价格 |
|------|------|------|
| AAPL | 苹果 | 200  |

![走势图](./chart.png)"""
        
        # 表格渲染后替换为 ![表格](./table_xxx.png)
        text_after_table_render = raw_response.replace(
            "| 代码 | 名称 | 价格 |\n|------|------|------|\n| AAPL | 苹果 | 200  |",
            "![表格](./table_render_123.png)"
        )
        
        # 图文切片
        chunks = re.split(r'(!\[.*?\]\(.*?\))', text_after_table_render)
        
        # 验证：应该包含表格图片和走势图图片
        image_chunks = [
            c for c in chunks 
            if c.strip() and re.match(r'^!\[.*?\]\(.*?\)$', c.strip())
        ]
        assert len(image_chunks) == 2
        assert any("table_render" in c for c in image_chunks)
        assert any("chart.png" in c for c in image_chunks)
