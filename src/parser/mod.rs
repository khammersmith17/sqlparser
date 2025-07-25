pub mod record;
use record::Record;
use record::Value;
use std::rc::Rc;

pub fn generate_condition_evaluator(condition: Option<Condition>) -> Box<dyn Fn(&Record) -> bool> {
    if let Some(cond) = condition {
        // closure that runs evaluate_record on the record and condition
        Box::new(move |row: &Record| {
            let Some(res) = cond.evaluate_record(row) else {
                return false;
            };
            res
        })
    } else {
        // closure the just returns true
        Box::new(|_row: &Record| true)
    }
}

#[derive(Debug, PartialEq)]
pub struct BasicSelectStatement<'a> {
    pub columns: Vec<&'a str>,
    pub table: &'a str,
    pub condition: Option<Condition>,
}

#[derive(Debug, PartialEq)]
pub enum Condition {
    Column(String),
    Value(Value),
    Operation {
        left: Box<Condition>,
        op: Operator,
        right: Box<Condition>,
    },
}

impl Condition {
    pub fn evaluate_record(&self, row: &Record) -> Option<bool> {
        // if left and right are Column/Value then evalute the condition
        // we should not get down to a Column/Value
        match self {
            Self::Operation { left, op, right } => match op {
                Operator::Or => Some(left.evaluate_record(row)? || right.evaluate_record(row)?),
                Operator::And => Some(left.evaluate_record(row)? && right.evaluate_record(row)?),
                _ => {
                    let left_v = left.resolve(row)?;
                    let right_v = right.resolve(row)?;
                    let result = Self::binary_op(left_v, op.to_owned(), right_v);
                    Some(result)
                }
            },
            _ => panic!("Invalid operation evaluation"),
        }
    }

    fn resolve(&self, row: &Record) -> Option<Value> {
        match self {
            Self::Column(c) => row.row_values.get(c).cloned(),
            Self::Value(v) => Some(v.clone()),
            Self::Operation { .. } => Some(Value::Bool(self.evaluate_record(row)?)),
        }
    }

    fn binary_op(lhs: Value, op: Operator, rhs: Value) -> bool {
        match (lhs, rhs) {
            (Value::Null, Value::Null) => true,
            (Value::Int(l), Value::Int(r)) => match op {
                Operator::Equal => l == r,
                Operator::NotEqual => l != r,
                Operator::LessThan => l < r,
                Operator::LessThanOrEqualTo => l <= r,
                Operator::GreaterThan => l > r,
                Operator::GreaterThanOrEqualTo => l >= r,
                _ => false,
            },
            (Value::Float(l), Value::Float(r)) => match op {
                Operator::Equal => l == r,
                Operator::NotEqual => l != r,
                Operator::LessThan => l < r,
                Operator::LessThanOrEqualTo => l <= r,
                Operator::GreaterThan => l > r,
                Operator::GreaterThanOrEqualTo => l >= r,
                _ => false,
            },
            (Value::String(l), Value::String(r)) => match op {
                Operator::Equal => l == r,
                Operator::NotEqual => l != r,
                _ => false,
            },
            _ => false,
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Operator {
    And,
    Or,
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    LessThanOrEqualTo,
    GreaterThanOrEqualTo,
}

// this is returned from a parser function
// it returns the part that has been parsed out and then the remaining portion of the text/tokens
type ParseResult<'a, Output> = Result<(&'a str, Output), String>;

// a trait with the function parse which takes in a str and returns some parsed str and additional
// tokens in output
pub trait Parser<'a, Output> {
    fn parse(&self, input: &'a str) -> ParseResult<'a, Output>;
}

// a container that holds a function pointer to a boxed parser function
pub struct BoxedParser<'a, Output> {
    f: Box<dyn Fn(&'a str) -> ParseResult<'a, Output> + 'a>,
}

// accept a parser function and box it
impl<'a, Output> BoxedParser<'a, Output> {
    fn new<F>(f: F) -> BoxedParser<'a, Output>
    where
        F: Fn(&'a str) -> ParseResult<'a, Output> + 'a,
    {
        BoxedParser { f: Box::new(f) }
    }
}

// take the boxed parser function and parse the input
impl<'a, Output> Parser<'a, Output> for BoxedParser<'a, Output> {
    fn parse(&self, input: &'a str) -> ParseResult<'a, Output> {
        // need to wrap f in () because it is a field with a function pointer and not a method on
        // the struct
        // in order to just for self.f(), f needs to be a method on the struct
        // in this case it is a function pointer in a field and not a function
        (self.f)(input)
    }
}

pub trait ParserExt<'a, Output>: Parser<'a, Output> + Sized {
    fn then<P2, Output2>(self, next: P2) -> BoxedParser<'a, (Output, Output2)>
    where
        P2: Parser<'a, Output2> + 'a,
        Output2: 'a,
        Output: 'a,
        Self: 'a,
    {
        BoxedParser::new(move |input| {
            self.parse(input).and_then(|(next_input, result)| {
                next.parse(next_input)
                    .map(|(final_input, next_result)| (final_input, (result, next_result)))
            })
        })
    }

    fn or<OtherParser>(self, other: OtherParser) -> BoxedParser<'a, Output>
    where
        OtherParser: Parser<'a, Output> + 'a,
        Self: 'a,
    {
        let p1 = Rc::new(self);
        let p2 = Rc::new(other);

        BoxedParser::new(move |input| match p1.parse(input) {
            Ok(res) => Ok(res),
            Err(_) => p2.parse(input),
        })
    }

    fn map<F, Output2>(self, f: F) -> BoxedParser<'a, Output2>
    where
        F: Fn(Output) -> Output2 + 'a,
        Output: 'a,
        Output2: 'a,
        Self: 'a,
    {
        BoxedParser::new(move |input| self.parse(input).map(|(rest, val)| (rest, f(val))))
    }
}

impl<'a, Output, P> ParserExt<'a, Output> for P where P: Parser<'a, Output> + Sized {}

// then define functions to return a BoxedParser
// define a closure then move that closure into a boxed parser
#[allow(dead_code)]
fn digit_char<'a>() -> impl Parser<'a, char> {
    BoxedParser::new(|input| {
        let mut chars = input.chars();
        if let Some(first) = chars.next() {
            if first.is_ascii_digit() {
                let rest = &input[first.len_utf8()..];
                return Ok((rest, first));
            }
        }
        Err("Expected digit".into())
    })
}

#[allow(dead_code)]
fn multi_digit_char<'a, Output, P>(parser: P) -> impl Parser<'a, Vec<Output>>
where
    P: Parser<'a, Output> + 'a,
    Output: 'a,
{
    BoxedParser::new(move |mut input| {
        let mut res: Vec<Output> = Vec::new();

        while let Ok((next, val)) = parser.parse(input) {
            res.push(val);
            input = next;
        }
        Ok((input, res))
    })
}

fn parse_for_select<'a>() -> impl Parser<'a, &'a str> {
    BoxedParser::new(|input| {
        let keyword = "select";

        let (head, rest) = input.split_at(keyword.len());

        if head.eq_ignore_ascii_case(head) {
            Ok((rest, keyword))
        } else {
            Err("Invalid query syntax".into())
        }
    })
}

fn parse_for_where<'a>() -> impl Parser<'a, &'a str> {
    BoxedParser::new(|input| {
        let keyword = "where";

        let (head, rest) = input.split_at(keyword.len());

        if head.eq_ignore_ascii_case(head) {
            Ok((rest, keyword))
        } else {
            Err("Invalid query syntax".into())
        }
    })
}

fn parse_for_int<'a>() -> impl Parser<'a, Condition> {
    BoxedParser::new(|input| {
        let mut i = 0_usize;
        for c in input.chars() {
            if c.is_numeric() {
                i += 1;
            } else {
                break;
            }
        }

        let value = &input[..i].parse::<i64>();
        match value {
            Ok(v) => Ok((&input[i..], Condition::Value(Value::Int(*v)))),
            Err(_) => Err("Not an integer".into()),
        }
    })
}

fn parse_for_float<'a>() -> impl Parser<'a, Condition> {
    // first parse out the whole number
    // then check for a .
    // then parse until not a number
    BoxedParser::new(|input| {
        let mut float_buffer = String::new();
        let mut chars = input.chars().peekable();
        while let Some(c) = chars.next_if(|&c| c.is_numeric()) {
            float_buffer.push(c);
        }

        let Some(decimal) = chars.next_if(|&c| c == '.') else {
            return Err("Not an float".into());
        };
        float_buffer.push(decimal);

        while let Some(c) = chars.next_if(|&c| c.is_numeric()) {
            float_buffer.push(c);
        }

        let value = float_buffer.parse::<f64>();
        let i = chars.collect::<Vec<_>>().len();
        match value {
            Ok(v) => Ok((&input[i..], Condition::Value(Value::Float(v)))),
            Err(_) => Err("Not an integer".into()),
        }
    })
}

fn parse_for_string_value<'a>() -> impl Parser<'a, Condition> {
    BoxedParser::new(|input| {
        let Some(first) = input.chars().nth(0) else {
            return Err("Equality value not a string".into());
        };

        if first != '\'' {
            return Err("Equality value not a string".into());
        }
        let Some(mut s) = input.find('\'') else {
            return Err("equality value not a string".into());
        };
        s += 1;
        let Some(e) = input[s..].find('\'') else {
            return Err("equality value not a string".into());
        };

        Ok((
            &input[s + e + 1..],
            Condition::Value(Value::String(input[s..s + e].to_string())),
        ))
    })
}

fn parse_for_condition_column<'a>() -> impl Parser<'a, Condition> {
    BoxedParser::new(|input| {
        let identifier_parser = parse_for_identifiers();

        let Ok((next, res)) = identifier_parser.parse(input) else {
            return Err("not an identifer".into());
        };

        Ok((next, Condition::Column(res.into())))
    })
}

fn parse_for_from<'a>() -> impl Parser<'a, &'a str> {
    BoxedParser::new(|input| {
        let keyword = "from";

        let (head, rest) = input.split_at(keyword.len());

        if head.eq_ignore_ascii_case(head) {
            Ok((rest, keyword))
        } else {
            Err("Invalid query syntax".into())
        }
    })
}

// parse until we reach then end of a word
fn parse_for_identifiers<'a>() -> impl Parser<'a, &'a str> {
    BoxedParser::new(|input| {
        let mut end = 0_usize;

        for c in input.chars() {
            if c.is_ascii_alphanumeric() || c == '_' {
                end += 1
            } else {
                break;
            }
        }

        if end == 0 {
            Err("expected identifier".into())
        } else {
            Ok((&input[end..], &input[..end]))
        }
    })
}

// parse until we reach a space
// then return the remaining input and an empty ()
fn parse_for_whitespace<'a>() -> impl Parser<'a, ()> {
    BoxedParser::new(|input| {
        let len = input.len();
        let mut end = 0_usize;
        for c in input.bytes() {
            if c == 32 {
                end += 1
            } else {
                break;
            }
        }

        if end == len {
            Err("Expected a space".into())
        } else {
            Ok((&input[end..], ()))
        }
    })
}

fn parse_for_operator<'a>() -> impl Parser<'a, Operator> {
    BoxedParser::new(|input| {
        let mut chars = input.chars().peekable();
        let Some(first) = chars.next() else {
            return Err("String is empty".into());
        };
        match first {
            '=' => {
                // return Operator::Equals
                return Ok((&input[1..], Operator::Equal));
            }
            '!' => {
                // check to make sure that the next character is =
                let Some(next) = chars.next() else {
                    return Err("Invalid operator".into());
                };
                if !matches!(next, '=') {
                    return Err("Invalid operator".into());
                }
                return Ok((&input[2..], Operator::NotEqual));
            }
            '<' => {
                // check to make sure that the next character is =
                let Some(next) = chars.next() else {
                    return Ok((&input[1..], Operator::LessThan));
                };
                if !matches!(next, '=') {
                    return Ok((&input[1..], Operator::LessThan));
                }
                return Ok((&input[2..], Operator::LessThanOrEqualTo));
            }
            '>' => {
                // check to make sure that the next character is =
                let Some(next) = chars.next() else {
                    return Ok((&input[1..], Operator::GreaterThan));
                };
                if !matches!(next, '=') {
                    return Ok((&input[1..], Operator::GreaterThan));
                }
                return Ok((&input[2..], Operator::GreaterThanOrEqualTo));
            }
            _ => return Err("Invalid operator".into()),
        }
    })
}

fn fold_conditions(op: Operator, conditions: Vec<Condition>) -> Option<Condition> {
    let mut cond_iter = conditions.into_iter();

    let first = cond_iter.next()?;

    Some(cond_iter.fold(first, |left, right| Condition::Operation {
        left: Box::new(left),
        op,
        right: Box::new(right),
    }))
}

fn or_precedence<'a>() -> impl Parser<'a, Option<Condition>> {
    let and_parser = and_precedence();
    BoxedParser::new(move |input| {
        let input_len = input.len();
        let mut start = 0_usize;
        let mut end = input.to_uppercase().find(" OR ").unwrap_or(input_len);
        let mut ops = Vec::<Condition>::new();
        while start < input_len {
            let Ok((_, op)) = and_parser.parse(&input[start..end]) else {
                return Err("Invalid where condition or".into());
            };
            ops.push(op);
            start = std::cmp::min(end + 4, input_len);
            end = input.to_lowercase().find(" OR ").unwrap_or(input_len);
        }

        let folded_op = fold_conditions(Operator::Or, ops);
        Ok(("", folded_op))
    })
}

fn and_precedence<'a>() -> impl Parser<'a, Condition> {
    let binary_op_parser = binary_op_precedence();
    BoxedParser::new(move |input| {
        let input_len = input.len();
        let mut start = 0_usize;
        let mut end = input.to_uppercase().find(" AND ").unwrap_or(input_len);
        let mut ops = Vec::<Condition>::new();
        while start < input_len {
            let Ok((_, op)) = binary_op_parser.parse(&input[start..end]) else {
                return Err("Invalid where condition and".into());
            };
            ops.push(op);
            start = std::cmp::min(end + 5, input_len);
            end = input.to_lowercase().find(" AND ").unwrap_or(input_len);
        }

        let Some(folded_op) = fold_conditions(Operator::And, ops) else {
            return Err("No ops to fold".into());
        };
        Ok(("", folded_op))
    })
}

fn binary_op_precedence<'a>() -> impl Parser<'a, Condition> {
    // parse out left
    // parse out whitespace
    // parse out operator
    // parse out right
    let whitespace = parse_for_whitespace();

    BoxedParser::new(move |mut input| {
        let value_parser = parse_for_float()
            .or(parse_for_int())
            .or(parse_for_string_value())
            .or(parse_for_condition_column());

        let op_parser = parse_for_operator();
        if let Ok((next, _)) = whitespace.parse(input) {
            input = next;
        }
        let Ok((mut input, left)) = value_parser.parse(input) else {
            return Err("no left side value".into());
        };

        if let Ok((next, _)) = whitespace.parse(input) {
            input = next;
        }

        let Ok((mut input, op)) = op_parser.parse(input) else {
            return Err("no operator".into());
        };
        if let Ok((next, _)) = whitespace.parse(input) {
            input = next;
        }

        let Ok((input, right)) = value_parser.parse(input) else {
            return Err("no right side value".into());
        };
        Ok((
            input,
            Condition::Operation {
                left: Box::new(left),
                op,
                right: Box::new(right),
            },
        ))
    })
}

fn parse_for_condition<'a>() -> impl Parser<'a, Option<Condition>> {
    or_precedence()
}

fn sep_by_two_seperators<'a, P, S, W, Output>(
    item: P,
    whitespace: W, // seperator on whitespace
    seperator: S,  // seperator on comma
) -> impl Parser<'a, Vec<Output>>
where
    P: Parser<'a, Output> + 'a,
    S: Parser<'a, ()> + 'a,
    W: Parser<'a, ()> + 'a,
    Output: 'a,
{
    BoxedParser::new(move |mut input| {
        let mut res_items = Vec::new();

        // parse out the first identifier from the input
        match item.parse(input) {
            Ok((next, res)) => {
                res_items.push(res);
                input = next;
            }
            Err(_) => return Ok((input, res_items)),
        }

        // parse for seperator
        // then parse for whitespace
        // until it is exhausted
        while let Ok((next, _)) = seperator.parse(input) {
            input = next;

            match whitespace.parse(input) {
                Ok((next, _)) => {
                    input = next;
                }
                Err(_) => {}
            };
            match item.parse(input) {
                Ok((next, res)) => {
                    res_items.push(res);
                    input = next;
                }
                Err(e) => return Err(e),
            }
        }
        Ok((input, res_items))
    })
}

fn parse_for_columns<'a>() -> impl Parser<'a, Vec<&'a str>> {
    sep_by_two_seperators(
        parse_for_identifiers(),
        parse_for_whitespace(),
        BoxedParser::new(|input| {
            if input.starts_with(",") {
                return Ok((&input[1..], ()));
            } else {
                return Err("Expected comma".into());
            }
        }),
    )
}

pub fn basic_select_statement<'a>() -> impl Parser<'a, BasicSelectStatement<'a>> {
    parse_for_select()
        .then(parse_for_whitespace())
        .then(parse_for_columns())
        .then(parse_for_whitespace())
        .then(parse_for_from())
        .then(parse_for_whitespace())
        .then(parse_for_identifiers())
        .map(
            |((((((_, _), columns), _), _), _), table)| BasicSelectStatement {
                columns,
                table,
                condition: None,
            },
        )
}

fn parse_for_where_clause<'a>() -> impl Parser<'a, Option<Condition>> {
    parse_for_whitespace()
        .then(parse_for_where())
        .then(parse_for_condition())
        .map(|((_, _), condition)| condition)
        .or(BoxedParser::new(|input| Ok((input, None))))
}

pub fn select_with_where<'a>() -> impl Parser<'a, BasicSelectStatement<'a>> {
    parse_for_select()
        .then(parse_for_whitespace())
        .then(parse_for_columns())
        .then(parse_for_whitespace())
        .then(parse_for_from())
        .then(parse_for_whitespace())
        .then(parse_for_identifiers())
        .then(parse_for_where_clause())
        .map(
            |(((((((_, _), columns), _), _), _), table), condition)| BasicSelectStatement {
                columns,
                table,
                condition,
            },
        )
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_single() {
        let parser = digit_char();
        let res = parser.parse("1122aa");
        assert_eq!(res, Ok(("122aa", '1')))
    }

    #[test]
    fn test_multiple() {
        let digit_parser = digit_char();
        let mdp = multi_digit_char(digit_parser);
        let res = mdp.parse("11122aa");
        assert_eq!(res, Ok(("aa", vec!['1', '1', '1', '2', '2'])))
    }

    #[test]
    fn select_parser() {
        let q = "Select * from table";
        let parser = parse_for_select();
        let res = parser.parse(q).unwrap();
        assert_eq!(res, (" * from table", "select"))
    }

    #[test]
    fn parser_for_identifier_and_whitespace() {
        let parser = parse_for_identifiers();
        let res = parser.parse("word1 word2").unwrap();

        let wparser = parse_for_whitespace();

        let res2 = wparser.parse(res.0).unwrap();
        assert_eq!(res2, ("word2", ()))
    }

    #[test]
    fn test_chain() {
        let chain = parse_for_select()
            .then(parse_for_whitespace())
            .then(parse_for_columns())
            .then(parse_for_whitespace())
            .then(parse_for_from())
            .then(parse_for_whitespace())
            .then(parse_for_identifiers())
            .map(|input| println!("input: {:?}", input));
        let _ = chain.parse("select column from table");
    }

    #[test]
    fn test_whole_parser() {
        let parser = basic_select_statement();
        let res = parser.parse("select column from table").unwrap();
        let actual = BasicSelectStatement {
            columns: ["column"].to_vec(),
            table: "table",
            condition: None,
        };
        assert_eq!(res, ("", actual))
    }

    #[test]
    fn multiple_columns() {
        let parser = basic_select_statement();
        let res = parser.parse("select column1, column2 from table").unwrap();
        let actual = BasicSelectStatement {
            columns: ["column1", "column2"].to_vec(),
            table: "table",
            condition: None,
        };
        assert_eq!(res, ("", actual))
    }

    #[test]
    fn float_parser() {
        let parser = parse_for_float();
        let result = parser.parse("12.0 abc");
        let res = result.unwrap();
        assert_eq!(res, (" abc", Condition::Value(Value::Float(12.0))))
    }

    #[test]
    fn int_parser() {
        let parser = parse_for_int();
        let result = parser.parse("12 abc");
        let res = result.unwrap();
        assert_eq!(res, (" abc", Condition::Value(Value::Int(12))))
    }

    #[test]
    fn string_value_parser() {
        let parser = parse_for_string_value();
        let result = parser.parse("'Yellow' ");
        let res = result.unwrap();
        assert_eq!(
            res,
            (" ", Condition::Value(Value::String(String::from("Yellow"))))
        )
    }

    #[test]
    fn test_parse_for_operator_geoet() {
        let parser = parse_for_operator();
        let res = parser.parse(">= thing").unwrap();
        assert_eq!(res, (" thing", Operator::GreaterThanOrEqualTo));
    }

    #[test]
    fn test_parse_for_operator_leoet() {
        let parser = parse_for_operator();
        let res = parser.parse("<= thing").unwrap();
        assert_eq!(res, (" thing", Operator::LessThanOrEqualTo));
    }

    #[test]
    fn test_parse_for_operator_ne() {
        let parser = parse_for_operator();
        let res = parser.parse("!= thing").unwrap();
        assert_eq!(res, (" thing", Operator::NotEqual));
    }

    #[test]
    fn test_parse_for_operator_gt() {
        let parser = parse_for_operator();
        let res = parser.parse("> thing").unwrap();
        assert_eq!(res, (" thing", Operator::GreaterThan));
    }

    #[test]
    fn test_parse_for_operator_lt() {
        let parser = parse_for_operator();
        let res = parser.parse("< thing").unwrap();
        assert_eq!(res, (" thing", Operator::LessThan));
    }

    #[test]
    fn test_parse_for_operator_e() {
        let parser = parse_for_operator();
        let res = parser.parse("= thing").unwrap();
        assert_eq!(res, (" thing", Operator::Equal));
    }

    #[test]
    fn test_parse_for_condition_identifer() {
        let parser = parse_for_condition_column();
        let res = parser.parse("thing = ").unwrap();
        assert_eq!(res, (" = ", Condition::Column(String::from("thing"))));
    }

    #[test]
    fn test_basic_condition() {
        let condition_parser = parse_for_condition();
        let res = condition_parser.parse("value >= 42").unwrap();
        assert_eq!(
            res,
            (
                "",
                Some(Condition::Operation {
                    left: Box::new(Condition::Column(String::from("value"))),
                    op: Operator::GreaterThanOrEqualTo,
                    right: Box::new(Condition::Value(Value::Int(42)))
                })
            )
        )
    }

    #[test]
    fn test_condition_w_and() {
        let condition_parser = parse_for_condition();
        let res = condition_parser
            .parse("value >= 42 and value2 = 1")
            .unwrap();
        println!("with and statement {:?}", res);
    }

    #[test]
    fn test_select_with_where() {
        let parser = select_with_where();
        let res = parser
            .parse("select value, value2 from column where value < 42 and value2 > 30.0")
            .unwrap();
        println!("full select statement: {:?}", res);
    }

    #[test]
    fn test_condition_eval() {
        let condition = Condition::Operation {
            left: Box::new(Condition::Column(String::from("value"))),
            op: Operator::LessThan,
            right: Box::new(Condition::Value(Value::Int(42_i64))),
        };

        let mut row_values: std::collections::HashMap<String, Value> =
            std::collections::HashMap::new();
        row_values.insert(String::from("value"), Value::Int(50));
        let record = Record {
            record_id: 0,
            row_values,
        };

        let is_met = condition.evaluate_record(&record);
        assert_eq!(is_met, Some(false));
    }
}
