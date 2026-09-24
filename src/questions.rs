//! The question set. This is the core asset of the project, kept server-side so
//! the browser cannot tamper with it.
//!
//! Upstream (jev-chat-jarvis) recommends writing questions in English because
//! Jev is trained mainly on English. We use Chinese instead: measured on real
//! support conversations it returned 0.99-1.00 confidence with correct verdicts,
//! and Chinese keeps the prompts editable for the people who own them. Revisit
//! if accuracy on a labelled batch turns out worse than an English variant.

use serde_json::{json, Value};

/// Seven judgements answered in one request. They are evaluated in parallel and
/// cannot see each other's answers, so none of them may depend on another.
pub fn judgement_questions() -> Value {
    json!({
        "intent": {
            "type": "choice",
            "instructions": "对方最后这几条消息的真实意图是什么？",
            "criteria": {
                "ask_info": "询问信息或确认事实",
                "request_action": "希望你去做某件事",
                "push_decision": "催促你表态或做决定",
                "complain": "表达不满、抱怨或指责",
                "socialize": "寒暄、闲聊或维持关系",
                "sell": "推销、拉人或营销",
                "probe_money": "试探钱财、账号、隐私或索要转账",
                "other": "以上都不匹配"
            }
        },
        // First draft of this question asked for a "risk level" and came back at
        // 0.20 confidence with probability spread evenly across all four bands --
        // the bands were too abstract to separate. `pressure` scored 0.99 on the
        // same conversation because "is there pressure" has observable evidence.
        // So each band now names concrete things that must appear in the text.
        "danger": {
            "type": "score",
            "instructions": "这段对话触及了哪一类需要谨慎对待的内容？只看对话里实际出现的东西。",
            "criteria": [
                "只是交换信息或闲聊，没有出现承诺、钱财、权限、个人信息",
                "出现了时间安排或工作交付的约定，属于日常范围",
                "要求你承诺具体结果，或出现了金额、合同、账号权限、私人信息",
                "索要转账、验证码、账号密码，或用身份与时限逼你立刻照做"
            ]
        },
        "wants": {
            "type": "choice",
            "instructions": "对方最想从你这里得到什么？",
            "criteria": {
                "an_answer": "一个明确的答复或信息",
                "a_commitment": "一个承诺、时间点或表态",
                "an_action": "你实际去执行某件事",
                "emotional_response": "情绪上的回应、安慰或认同",
                "money_or_access": "钱、账号、权限或隐私信息",
                "nothing_specific": "没有具体诉求"
            }
        },
        "reply_now": {
            "type": "noul",
            "instructions": "现在是否应该立刻回复？",
            "criteria": {
                "true": "对方在等答复，拖延会造成误解或损失",
                "false": "可以晚点回，或需要先想清楚、先去核实、甚至不该回"
            }
        },
        // The first version used `answer_directly: "直接给出答复"`, which reads
        // both as "do what they asked" and as "reply explaining your situation".
        // Five of six drafting models took the second reading and wrote refusals
        // even when the verdict said answer_directly. It also had no option for
        // partial compliance, so a conversation like "send me the draft as-is"
        // had to be forced into either full compliance or stalling — which is
        // part of why confidence sat at 0.54. Each option now states what you
        // actually do, and the boundaries between them are explicit.
        "best_action": {
            "type": "choice",
            "instructions": "此刻最合适的动作是什么？只看动作本身，不考虑措辞。",
            "criteria": {
                "comply": "照对方的要求去做，或把对方索要的信息直接给出",
                "comply_partly": "先给出手上能给的部分，同时说明还缺什么",
                "ask_clarify": "先反问，把对方的诉求或前提问清楚，这一轮不给结论",
                "buy_time": "只回应一句表示收到，把实际处理推到之后",
                "decline": "明确拒绝，或划清界限说明这件事不能做",
                "escalate": "交给别人处理，或换更正式的渠道留痕",
                "ignore": "不回复，不要接这个话头"
            }
        },
        // This used to ask "what tone should the reply use", which scored 0.17 --
        // it is a recommendation, and recommendations have no observable evidence
        // in the transcript. Asking about the counterpart's own style instead
        // turns it into an observation, which is what Jev is good at. The drafting
        // model then decides how to respond to that style.
        "counterpart_style": {
            "type": "choice",
            "instructions": "对方在这段对话里的说话风格是哪一种？描述对方，不是建议你怎么回。",
            "criteria": {
                "casual": "口语化短句，有语气词、表情或玩笑，像熟人聊天",
                "businesslike": "直接说事、用词规范，典型的工作沟通口吻",
                "terse": "极简，只给结论或指令，不解释不铺垫",
                "emotional": "带明显情绪，出现抱怨、指责、反问或急切的表达"
            }
        },
        "pressure": {
            "type": "score",
            "instructions": "对方施加的压力有多大？",
            "criteria": [
                "没有施压，正常交流",
                "有催促或期待，但可以商量",
                "强烈施压，用情绪或时限逼你表态"
            ]
        }
    })
}

/// Ranking is a second Jev call: which of the three drafts fits best.
pub fn ranking_questions(count: usize) -> Value {
    let mut criteria = serde_json::Map::new();
    for index in 0..count {
        criteria.insert(
            format!("draft_{}", index + 1),
            json!(format!("第 {} 条候选回复", index + 1)),
        );
    }
    json!({
        "best_draft": {
            "type": "choice",
            "instructions": "综合判断结果和对话上下文，哪一条候选回复最合适？",
            "criteria": Value::Object(criteria)
        }
    })
}

/// The drafting model is a normal chat LLM. It must not invent facts, and must
/// respect the judgement Jev already produced.
pub const DRAFT_SYSTEM_PROMPT: &str = r#"你在帮用户起草聊天回复。用户会给你一段对话、以及一个判断模型对这段对话的分析结果。

请起草 3 条候选回复，要求：
- 口语化，像真人在聊天，不要写成公文或客服话术
- 长度控制在 1-3 句，除非对话明显需要更长
- 严格执行分析结果里的 best_action（最佳动作），它决定你这条回复要做什么。特别注意：comply 是真的照对方要求去做，不是回一句解释为什么做不了；comply_partly 是先给出部分成果；buy_time 才是只应一声把事推后
- 参照 counterpart_style（对方的说话风格）调整用词的松紧：对方随意你就别太正式，对方极简你也别长篇大论；但如果 danger 较高或 best_action 是拒绝，措辞要比对方更清楚、更留得住底
- 三条之间要有真实差异：可以是详略不同、态度软硬不同、或给不给承诺不同
- 绝对不要编造对话和背景信息里没有的事实、数字、时间点或承诺
- 如果分析结果建议拒绝或不回，那么候选里应体现拒绝或极简回应，不要强行热情

输出格式：一个 JSON 对象，只含 drafts 一个键，值是长度为 3 的字符串数组，每个字符串就是一条可以直接发出去的回复原文。

务必注意：数组里放的是真正写好的回复内容，不是占位符。不要输出 "..."、"第一条"、"候选回复1" 这类示意文字。回复用对话本身的语言书写。"#;

/// Splitting a pasted transcript. Unlike a screenshot this has no bubble
/// geometry, so speaker attribution is not merely hard — it is absent from the
/// input. A WeChat copy labels the user's own messages with their nickname, not
/// with "me", so nothing in the text says which name belongs to the reader.
///
/// The model is therefore told not to guess: every turn comes back as `other`
/// and the UI asks the user to point at their own name. A guess that lands wrong
/// inverts the whole downstream pipeline (`intent` asks about the *other* party,
/// `best_action` about what *you* should do), and it would look plausible enough
/// that the user would not check it. One click beats a confident wrong answer.
pub const IMPORT_SYSTEM_PROMPT: &str = r#"你在读一段从聊天软件里复制出来的文字记录，任务是把它切分成结构化的消息列表。

切分规则：
- 一条消息通常由三部分组成：发送者昵称、时间、正文。三者可能各占一行，中间可能有空行，也可能昵称和时间挤在同一行
- 发送者昵称写进 name，正文写进 text
- 时间戳全部丢弃，不要留在 text 里。常见形式：2026年09月24日 7:40、2026/09/24 15:07、昨天 14:23、上午 10:05、15:09
- 日期分隔线、「以下为新消息」、「XX撤回了一条消息」、「对方正在输入」、「已读」这类系统提示全部丢弃
- 同一个人连续发的多条消息，必须各自独立成条，绝对不要合并。连发几条本身就是信息
- 一条消息内部的换行保留为 \n
- [强] [图片] [语音] [表情] [链接] [文件] 这类方括号占位原样抄录，不要改写成文字描述
- @某人 保留在正文里
- 引用或回复某条消息的引用块，写进该条 text 的开头，用「引用：」标明
- 严格保留原文用字，包括错别字、省略号、标点误用，不要润色

关于 speaker：
- speaker 一律填 "other"
- 不要试图判断哪个昵称是当前用户。复制出来的记录里，用户自己发的消息显示的也是昵称，文字本身没有这个信息。归属由用户在界面上指定，你猜错会让后续判断整体反向

把下面这些情况的序号（从 1 开始）记进 uncertain：
- 没有昵称，无法确定是谁说的
- 正文明显被截断或看起来不完整
- 你不确定某一段到底是一条消息还是两条

顺序按原文从上到下。

只输出 JSON：
{"conversation":[{"speaker":"other","name":"沐桐旭","text":"..."}],"uncertain":[3],"note":""}

note 里写一句你对这段记录的判断（单聊还是群聊、识别到几个说话人、有没有明显缺失），没有就留空。"#;

/// Reading a chat screenshot. The hard part is not the characters, it is telling
/// who said what: that information lives in bubble position and colour, which is
/// why this goes to a vision model rather than a plain OCR engine.
pub const VISION_SYSTEM_PROMPT: &str = r#"你在读一张聊天软件的截图，任务是把屏幕上的对话转成结构化数据。

判断说话人：
- 气泡靠右、或底色是绿色/蓝色等主题色的，是「我」（speaker 填 me）
- 气泡靠左、或底色是白色/灰色的，是「对方」（speaker 填 other）
- 位置和颜色冲突时以位置为准，因为不同软件配色不同
- 实在判断不出来的，填 other 并在 uncertain 里记下该条的序号

抄录文字：
- 严格保留原文，不要改写、润色、翻译或补全
- 不要抄时间戳、日期分隔线、「已读」「已送达」标记、头像上的昵称
- 群聊里气泡上方的发送者昵称，写进该条的 name 字段
- 一条消息内的换行保留为 \n
- 被截断看不全的消息，照抄可见部分，并把序号记进 uncertain
- 图片、表情包、语音、文件等非文字消息，text 写成 [图片] [语音] [文件：名称] 这样的占位

顺序按屏幕从上到下。

只输出 JSON：
{"conversation":[{"speaker":"other","name":"","text":"..."}],"uncertain":[2],"note":""}

note 里可以写一句你对这张图的判断说明（比如软件名、是群聊还是单聊、有没有明显遮挡），没有就留空。"#;
